//! The issue-report set (`docs/s6-secrets2-prereg.md`, section 3.2; Zenodo
//! 10.5281/zenodo.19622962, CC BY 4.0): its train and validation splits as candidate records.
//!
//! A row is a candidate string, the issue text it was matched in, and a 0/1 label. The candidate
//! is located in its text (its first occurrence) and written as a candidate record: the line of
//! its first character, the neighbours, no path. The set says nothing of a rule, a name or the
//! engine's filters, so the record has the rule id [`RULE`], no name, and no filter flag.
//!
//! Two things happen before a record is written:
//! - **decontamination**: a train or validation row whose text is also the text of a test row is
//!   removed. The test split is read for this and nothing else: of each of its rows only a hash
//!   of the text is kept;
//! - **replacement**: every positive candidate string of a train or validation row is replaced,
//!   wherever it stands in any text of these two splits, by its word-preserving shape
//!   replacement ([`super::replace`]). Records are cut from the replaced texts, so the real
//!   value of a positive is in no record: neither as a value nor in a line of another
//!   candidate, of the same issue or of another. The build checks it on what it wrote.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde_json::json;
use sky_infer::secret_cand::WordSet;

use super::replace::{replace_value_report, KeptRuns};
use super::{
    bump, file_entry, record_at, write_json, CandRecord, Counts, Item, Label, RecFilters, ISSUES,
};
use crate::csv::parse_csv;
use crate::record::write_jsonl;

/// The rule id of an issue-report candidate: the set's own regular expressions found it.
pub const RULE: &str = "issue-regex";
/// The `generator` field of its records.
pub const GENERATOR: &str = "issue-report";

/// Where the split files are, below the data folder.
pub const DATA_DIR: &str = "issues/unzipped/Secret-Leak-Detection-Issue-Report/Data";

/// One row of a split file.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    /// 0-based index of the row in its file.
    pub index: usize,
    pub candidate: String,
    pub text: String,
    pub positive: bool,
}

/// The rows of a split file (`candidate_string`, `text`, `label` by their header names).
pub fn read_rows(path: &Path) -> Result<Vec<Row>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut rows = parse_csv(&text).into_iter();
    let header = rows.next().context("empty split file")?;
    let col = |name: &str| {
        header
            .iter()
            .position(|h| h == name)
            .with_context(|| format!("{}: no column {name}", path.display()))
    };
    let (c, t, l) = (col("candidate_string")?, col("text")?, col("label")?);
    rows.enumerate()
        .map(|(index, mut r)| {
            if r.len() != header.len() {
                bail!("{}: row {index} has {} fields", path.display(), r.len());
            }
            let positive = match r[l].trim() {
                "1" => true,
                "0" => false,
                other => bail!("{}: row {index} has label {other:?}", path.display()),
            };
            Ok(Row {
                index,
                candidate: std::mem::take(&mut r[c]),
                text: std::mem::take(&mut r[t]),
                positive,
            })
        })
        .collect()
}

/// The hashes of the texts of a split file. This is all the build reads of the test split.
pub fn text_hashes(path: &Path) -> Result<(HashSet<[u8; 32]>, usize)> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut rows = parse_csv(&text).into_iter();
    let header = rows.next().context("empty split file")?;
    let t = header
        .iter()
        .position(|h| h == "text")
        .with_context(|| format!("{}: no column text", path.display()))?;
    let mut set = HashSet::new();
    let mut n = 0;
    for r in rows {
        n += 1;
        set.insert(hash(r.get(t).map_or("", String::as_str)));
    }
    Ok((set, n))
}

fn hash(text: &str) -> [u8; 32] {
    let d = ring::digest::digest(&ring::digest::SHA256, text.as_bytes());
    let mut out = [0u8; 32];
    out.copy_from_slice(d.as_ref());
    out
}

/// The positive candidate strings and their replacements, indexed by their first bytes so that
/// every text can be searched for all of them in one pass.
pub struct Positives {
    /// Bytes of the index key: the length of the shortest string, 8 at most.
    key: usize,
    by_prefix: HashMap<Vec<u8>, Vec<usize>>,
    /// `(string, replacement)`, the replacement of the same length in bytes.
    pairs: Vec<(String, String)>,
}

impl Positives {
    pub fn new(replacement: &HashMap<String, String>) -> Self {
        let mut pairs: Vec<(String, String)> = replacement
            .iter()
            // a string that stays as it is (kept words and separators only) is not replaced
            .filter(|(s, new)| !s.is_empty() && s.len() == new.len() && s != new)
            .map(|(s, new)| (s.clone(), new.clone()))
            .collect();
        pairs.sort();
        let key = pairs.iter().map(|p| p.0.len()).min().unwrap_or(1).min(8);
        let mut by_prefix: HashMap<Vec<u8>, Vec<usize>> = HashMap::new();
        for (i, (s, _)) in pairs.iter().enumerate() {
            by_prefix
                .entry(s.as_bytes()[..key].to_vec())
                .or_default()
                .push(i);
        }
        Self {
            key,
            by_prefix,
            pairs,
        }
    }

    /// Every occurrence of a positive string in `text`: `(start, index of the pair)`.
    fn occurrences(&self, text: &str) -> Vec<(usize, usize)> {
        let b = text.as_bytes();
        let mut out = Vec::new();
        if self.pairs.is_empty() || b.len() < self.key {
            return out;
        }
        for at in 0..=b.len() - self.key {
            if let Some(list) = self.by_prefix.get(&b[at..at + self.key]) {
                for &i in list {
                    if b[at..].starts_with(self.pairs[i].0.as_bytes()) {
                        out.push((at, i));
                    }
                }
            }
        }
        out
    }

    /// Whether `text` holds a positive string as it was published.
    pub fn any_in(&self, text: &str) -> bool {
        !self.occurrences(text).is_empty()
    }

    /// `text` with every occurrence of every positive string replaced by its replacement.
    /// Longer strings are written first, so where two overlap the shorter one's replacement
    /// stands inside the longer one's: no character of an occurrence keeps its original unless
    /// a replacement keeps it. `None` when the text holds none.
    pub fn replace_in(&self, text: &str) -> Option<String> {
        let mut found = self.occurrences(text);
        if found.is_empty() {
            return None;
        }
        found.sort_unstable_by(|a, b| {
            let (la, lb) = (self.pairs[a.1].0.len(), self.pairs[b.1].0.len());
            lb.cmp(&la).then(a.cmp(b))
        });
        let mut out = text.as_bytes().to_vec();
        for (at, i) in found {
            let new = self.pairs[i].1.as_bytes();
            out[at..at + new.len()].copy_from_slice(new);
        }
        // replacements copy every non-ASCII byte in place: the text stays valid UTF-8
        Some(String::from_utf8(out).unwrap_or_else(|_| text.to_string()))
    }
}

/// The candidate record of a candidate string in its text: its first occurrence. `None` when
/// the text does not hold it.
pub fn candidate_record(text: &str, candidate: &str) -> Option<CandRecord> {
    if candidate.is_empty() {
        return None;
    }
    let at = text.find(candidate)?;
    Some(record_at(
        GENERATOR,
        None,
        RULE,
        None,
        text,
        at,
        at + candidate.len(),
        RecFilters::default(),
    ))
}

pub struct IssueOptions {
    /// The data folder (it holds `issues/`).
    pub data: PathBuf,
    /// The build folder.
    pub out: PathBuf,
    pub seed: u64,
    pub words: WordSet,
}

/// Builds `<out>/candidates/issues-{train,val}.jsonl` and `<out>/counts/issues.json`.
pub fn build(o: &IssueOptions) -> Result<serde_json::Value> {
    let dir = o.data.join(DATA_DIR);
    let files = [
        ("train", dir.join("train.csv")),
        ("val", dir.join("val.csv")),
    ];
    let test = dir.join("test.csv");
    let (test_texts, test_rows) = text_hashes(&test)?;
    let mut inputs = serde_json::Map::new();
    for (name, path) in files.iter().map(|(n, p)| (*n, p)).chain([("test", &test)]) {
        inputs.insert(format!("{name}.csv"), file_entry(path)?);
    }

    // rows that stay after the decontamination
    let mut kept: Vec<(&str, Vec<Row>)> = Vec::new();
    let mut counts = serde_json::Map::new();
    for (split, path) in &files {
        let rows = read_rows(path)?;
        let mut c = Counts::new();
        c.insert("rows".into(), rows.len());
        c.insert(
            "positives".into(),
            rows.iter().filter(|r| r.positive).count(),
        );
        let (mut removed, mut removed_pos) = (0, 0);
        let rows: Vec<Row> = rows
            .into_iter()
            .filter(|r| {
                let dup = test_texts.contains(&hash(&r.text));
                removed += dup as usize;
                removed_pos += (dup && r.positive) as usize;
                !dup
            })
            .collect();
        c.insert("removed_text_in_test".into(), removed);
        c.insert("removed_text_in_test_positives".into(), removed_pos);
        counts.insert(split.to_string(), json!(c));
        kept.push((split, rows));
    }

    // the context runs (amendments 4, 5 and 7): the runs of letters of at least five distinct
    // negative candidate strings of the training split, after its decontamination, that are a
    // dictionary word, hold a keyword, or have four letters that are not all hexadecimal
    let negatives: HashSet<&str> = kept
        .iter()
        .filter(|(split, _)| *split == "train")
        .flat_map(|(_, rows)| rows.iter().filter(|r| !r.positive))
        .map(|r| r.candidate.as_str())
        .collect();
    let keep = KeptRuns::from_negatives(negatives.iter().copied(), &o.words);

    // one replacement per positive candidate string, the same in both splits
    let mut replacement: HashMap<String, String> = HashMap::new();
    let mut unchanged = 0usize;
    let (mut with_replaced, mut with_kept, mut kept_runs) = (0usize, 0usize, 0usize);
    let mut with_redrawn_common = 0usize;
    let mut redrawn_words: BTreeMap<String, usize> = BTreeMap::new();
    let mut kept_words: BTreeMap<String, usize> = BTreeMap::new();
    for (_, rows) in &kept {
        for r in rows.iter().filter(|r| r.positive) {
            if replacement.contains_key(&r.candidate) {
                continue;
            }
            let new = replace_value_report(&r.candidate, &o.words, &keep, o.seed);
            unchanged += (new.text == r.candidate) as usize;
            with_replaced += (new.replaced > 0) as usize;
            with_kept += !new.kept.is_empty() as usize;
            with_redrawn_common += !new.redrawn_common.is_empty() as usize;
            for w in new.redrawn_common.iter().collect::<HashSet<_>>() {
                *redrawn_words.entry(w.clone()).or_default() += 1;
            }
            kept_runs += new.kept.len();
            // a word counts once for each distinct positive string that holds it
            for w in new.kept.iter().collect::<HashSet<_>>() {
                *kept_words.entry(w.clone()).or_default() += 1;
            }
            replacement.insert(r.candidate.clone(), new.text);
        }
    }
    let mut most: Vec<(&String, &usize)> = kept_words.iter().collect();
    most.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    let top = |most: Vec<(&String, &usize)>| -> Vec<serde_json::Value> {
        most.iter()
            .take(10)
            .map(|(w, n)| json!({"run": w, "positive_strings": n}))
            .collect()
    };
    let most = top(most);
    let mut redrawn: Vec<(&String, &usize)> = redrawn_words.iter().collect();
    redrawn.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    let redrawn = top(redrawn);
    let positives = Positives::new(&replacement);

    let mut left_in_records = 0usize;
    for (split, rows) in &kept {
        let mut items = Vec::with_capacity(rows.len());
        let mut c = Counts::new();
        for r in rows {
            // located in the text as published, cut from the text as replaced
            let Some(at) = (!r.candidate.is_empty())
                .then(|| r.text.find(&r.candidate))
                .flatten()
            else {
                bump(&mut c, "not_found");
                continue;
            };
            if r.text[at + 1..].contains(&r.candidate) {
                bump(&mut c, "several_occurrences_first_taken");
            }
            let replaced = positives.replace_in(&r.text);
            if replaced.is_some() {
                bump(&mut c, "texts_with_a_replacement");
            }
            let rec = record_at(
                GENERATOR,
                None,
                RULE,
                None,
                replaced.as_deref().unwrap_or(&r.text),
                at,
                at + r.candidate.len(),
                RecFilters::default(),
            );
            if r.positive {
                bump(&mut c, "positives_replaced");
            } else if rec.value != r.candidate {
                bump(&mut c, "negatives_whose_value_overlaps_a_replaced_positive");
            }
            if rec.end_line > rec.line {
                bump(&mut c, "over_several_lines");
            }
            // what the record holds of the text: no positive string as it was published
            let fields = [
                Some(rec.value.as_str()),
                Some(rec.line_text.as_str()),
                rec.prev_line.as_deref(),
                rec.next_line.as_deref(),
            ];
            left_in_records += fields
                .iter()
                .flatten()
                .filter(|f| positives.any_in(f))
                .count();
            bump(&mut c, if r.positive { "T" } else { "F" });
            items.push(Item {
                id: format!("issue:{split}:{}", r.index),
                source: ISSUES.to_string(),
                split: split.to_string(),
                label: if r.positive { Label::T } else { Label::F },
                weak: false,
                kind: false,
                class: None,
                via: None,
                rec,
            });
        }
        c.insert("candidates".into(), items.len());
        let mut merged: BTreeMap<String, serde_json::Value> =
            serde_json::from_value(counts[*split].clone())?;
        for (k, v) in c {
            merged.insert(k, json!(v));
        }
        counts.insert(split.to_string(), json!(merged));
        write_jsonl(
            &o.out.join(format!("candidates/issues-{split}.jsonl")),
            items.iter(),
        )?;
    }
    if left_in_records > 0 {
        bail!("{left_in_records} record fields still hold a positive value as published");
    }

    let out = json!({
        "source": "Zenodo 10.5281/zenodo.19622962, version 19622962 (CC BY 4.0)",
        "inputs": inputs,
        "test_rows_hashed_for_decontamination": test_rows,
        "test_distinct_texts": test_texts.len(),
        "rule": RULE,
        "replacement": {
            "distinct_positive_strings": replacement.len(),
            "unchanged_made_of_kept_words_and_separators": unchanged,
            "distinct_negative_strings_of_the_training_split": negatives.len(),
            "kept_runs_of_the_training_negatives": keep.len(),
            "positive_strings_with_a_replaced_word_run": with_replaced,
            "positive_strings_with_a_redrawn_run_that_5_negatives_hold": with_redrawn_common,
            "ten_most_frequent_redrawn_runs_that_5_negatives_hold": redrawn,
            "positive_strings_with_a_kept_run": with_kept,
            "kept_runs_in_the_positive_strings": kept_runs,
            "distinct_kept_runs_in_the_positive_strings": kept_words.len(),
            "ten_most_frequent_kept_runs": most,
            "published_positive_strings_left_in_a_record": left_in_records,
        },
        "splits": counts,
    });
    write_json(&o.out.join("counts/issues.json"), &out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secrets2::replace::replace_value;

    fn words() -> WordSet {
        WordSet::from_text("winter\nsummer\nspring\n")
    }

    #[test]
    fn a_candidate_is_located_and_cut_like_a_generator_record() {
        let text = "Steps:\r\n1. run it\r\n  token = abc123XYZ\r\nthen it fails\r\n";
        let r = candidate_record(text, "abc123XYZ").unwrap();
        assert_eq!((r.rule.as_str(), r.generator.as_str()), (RULE, GENERATOR));
        assert_eq!((r.line, r.end_line, r.start, r.end), (3, 3, 10, 19));
        assert_eq!(r.line_text, "  token = abc123XYZ");
        assert_eq!(r.prev_line.as_deref(), Some("1. run it"));
        assert_eq!(r.next_line.as_deref(), Some("then it fails"));
        assert_eq!(
            (r.file.as_deref(), r.ext.as_str(), r.name.as_deref()),
            (None, "", None)
        );
        assert!(candidate_record(text, "missing").is_none());
        assert!(candidate_record(text, "").is_none());
        // the first of several occurrences
        let r = candidate_record("x=K9 and again K9", "K9").unwrap();
        assert_eq!(r.start, 2);
        // a match over two lines: the first line, and the line after the last
        let r = candidate_record("a\nkey: v1\nmore v2\nz", "v1\nmore v2").unwrap();
        assert_eq!((r.line, r.end_line), (2, 3));
        assert_eq!(r.line_text, "key: v1");
        assert_eq!(r.next_line.as_deref(), Some("z"));
    }

    #[test]
    fn every_occurrence_of_a_positive_is_replaced_in_every_text() {
        let w = words();
        let secret = "Winter-48213-qzv";
        let text = format!("first {secret}\nagain: {secret} and a neighbour 77\n");
        let mut rep = HashMap::new();
        rep.insert(
            secret.to_string(),
            replace_value(secret, &w, &KeptRuns::default(), 1234),
        );
        let p = Positives::new(&rep);
        let out = p.replace_in(&text).unwrap();
        assert_eq!(out.len(), text.len());
        assert!(!out.contains(secret) && p.any_in(&text) && !p.any_in(&out));
        assert_eq!(out.matches(rep[secret].as_str()).count(), 2);
        // the rest of the text is untouched, and a text without the value is left alone
        assert!(out.starts_with("first ") && out.ends_with(" and a neighbour 77\n"));
        assert!(p.replace_in("another issue, nothing here").is_none());
        // the word stays a word, in its place and case
        let new = &rep[secret];
        assert!(w.is_word(&new[..6]) && new.starts_with(|c: char| c.is_ascii_uppercase()));
        // a shorter positive inside a longer one, and two that overlap: nothing is left
        let (long, short, other) = (
            "pw=Qzvmer-99310042-xk",
            "Qzvmer-99310042",
            "99310042-xk-Hjklng-7",
        );
        let text = format!("{long} | {short} | pw=Qzvmer-{other}");
        let mut rep = HashMap::new();
        for s in [long, short, other] {
            rep.insert(
                s.to_string(),
                replace_value(s, &w, &KeptRuns::default(), 1234),
            );
        }
        let p = Positives::new(&rep);
        let out = p.replace_in(&text).unwrap();
        assert!(!out.contains("99310042") && !out.contains("Qzvmer") && !out.contains("Hjklng"));
        assert_eq!(out.len(), text.len());
        assert!(!p.any_in(&out));
        // non-ASCII text around a value stays as it is
        let text = format!("cl\u{e9} : {secret} \u{1F512}");
        let mut rep = HashMap::new();
        rep.insert(
            secret.to_string(),
            replace_value(secret, &w, &KeptRuns::default(), 1234),
        );
        let out = Positives::new(&rep).replace_in(&text).unwrap();
        assert!(out.starts_with("cl\u{e9} : ") && out.ends_with(" \u{1F512}"));
    }

    #[test]
    fn rows_are_read_by_header_name() {
        let dir = std::env::temp_dir().join(format!("sky-issues-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("s.csv");
        std::fs::write(
            &p,
            ",Unnamed: 0,candidate_string,text,Issue_id,label\n0,5,abc,\"see abc\nhere\",u1,1\n1,9,\"x,y\",\"a \"\"x,y\"\" b\",u2,0\n",
        )
        .unwrap();
        let rows = read_rows(&p).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(
            (rows[0].candidate.as_str(), rows[0].positive),
            ("abc", true)
        );
        assert_eq!(rows[0].text, "see abc\nhere");
        assert_eq!((rows[1].candidate.as_str(), rows[1].index), ("x,y", 1));
        assert_eq!(rows[1].text, "a \"x,y\" b");
        let (set, n) = text_hashes(&p).unwrap();
        assert_eq!((set.len(), n), (2, 2));
        assert!(set.contains(&hash("see abc\nhere")));
        std::fs::remove_dir_all(dir).ok();
    }
}
