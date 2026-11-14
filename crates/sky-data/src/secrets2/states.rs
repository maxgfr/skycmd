//! The states of the candidates of a split, for the evaluation
//! (`docs/s6-secrets2-prereg.md`, sections 6 and 7; `scripts/secrets2-eval.sh`):
//! `sky-data secrets2 states`.
//!
//! The model and the size-matched logistic regression judge every candidate of a split, not
//! only the labelled ones the training build keeps. This command turns candidates into the
//! table of states and the decision records the evaluation reads, with the state builder the
//! training data was built with ([`super::record_state`]):
//!
//! | corpus | candidates | ids |
//! |---|---|---|
//! | `creddata:<split>` | the generator's raw export on the split's repositories | `cred:<train\|val\|test>:<index>` |
//! | `clean` | the generator's raw export on the clean-text corpus | `clean:all:<index>` |
//! | `issues:<split>` | the dataset's candidates, in the issue texts as published | `issue:<train\|val\|test>:<row>` |
//! | `issues:<split>:obfuscated` | the same, in the texts passed through CredData's obfuscation | `issue:…:<row>` |
//! | `issues:<split>:e2e` | the generator's raw export on the issue texts (end to end) | `issue-e2e:…:<index>` |
//!
//! The index is the candidate's line in the export; the row is the row of the split file. For
//! the issue reports the command reads what `scripts/secrets2/issues.py prepare` (and
//! `obfuscate`) wrote under `<data>/eval/issues/<split>`: the texts as files and the place of
//! each candidate in its text. Nothing is replaced here: these are the values as published, or
//! as CredData's function made them.
//!
//! It writes `<out>.jsonl` (the table: id, state, features, one line per candidate, in the
//! order of the input) and `<out>.records.jsonl` (the decision records, same order): the real
//! question for every candidate, then the kind question for a CredData candidate whose label is
//! known. A CredData candidate takes its label from the meta rows by the matching rules of the
//! training build ([`super::cred::FileRows`]); an issue-report candidate from its row. The
//! others have no label: their record's target is a placeholder (`false`), and their source
//! says so (`creddata_unlabelled:real`, `clean:real`, `issues_e2e:real`). The `split` of every
//! record is `val`, whatever the corpus: it is the name `sky-eval` is asked for.
//!
//! **The test split is refused** unless the guard of the evaluation's entry point is satisfied
//! ([`test_is_open`]): the file `runs/s6-secrets2/TEST-OPEN` exists, and a model file is given
//! whose metadata holds its threshold (`skycmd-s2.secrets2.tau_f1`, stored before any test file
//! is read). A candidate of a repository of another split stops the run.

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde_json::json;
use sky_infer::secret_cand::{self as sc, Features, WordSet, KINDS, Q_KIND, Q_REAL};
use sky_infer::tokenizer::Tokenizer;

use super::cred::{read_meta_eval, read_utf16, repo_of, FileRows, MetaRow, Own, Splits};
use super::mix::features_json;
use super::{
    bump, file_entry, issues, read_json, record_at, record_state, stored_scope, CandRecord, Counts,
    Label, RecFilters,
};
use crate::record::{one_hot, Question};

/// The metadata key of the operating threshold in the model's GGUF (section 5).
pub const TAU_KEY: &str = "skycmd-s2.secrets2.tau_f1";

/// The file whose existence opens the test split, below the checkout.
pub const TEST_OPEN: &str = "runs/s6-secrets2/TEST-OPEN";

/// Which candidates of the issue reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Published,
    Obfuscated,
    EndToEnd,
}

/// A set of candidates the evaluation asks states for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Corpus {
    CredData { split: String },
    Clean,
    Issues { split: String, variant: Variant },
}

impl Corpus {
    /// `creddata:<split>`, `clean`, `issues:<split>`, `issues:<split>:obfuscated` or
    /// `issues:<split>:e2e`, the split being `training`, `validation` or `test`.
    pub fn parse(name: &str) -> Result<Self> {
        let parts: Vec<&str> = name.split(':').collect();
        let split = |s: &str| -> Result<String> {
            if !["training", "validation", "test"].contains(&s) {
                bail!("unknown split {s:?} (training, validation, test)");
            }
            Ok(s.to_string())
        };
        Ok(match parts.as_slice() {
            ["clean"] => Corpus::Clean,
            ["creddata", s] => Corpus::CredData { split: split(s)? },
            ["issues", s] => Corpus::Issues {
                split: split(s)?,
                variant: Variant::Published,
            },
            ["issues", s, "obfuscated"] => Corpus::Issues {
                split: split(s)?,
                variant: Variant::Obfuscated,
            },
            ["issues", s, "e2e"] => Corpus::Issues {
                split: split(s)?,
                variant: Variant::EndToEnd,
            },
            _ => bail!("unknown corpus {name:?}"),
        })
    }

    pub fn split(&self) -> Option<&str> {
        match self {
            Corpus::Clean => None,
            Corpus::CredData { split } | Corpus::Issues { split, .. } => Some(split),
        }
    }

    /// `train`, `val` or `test`: the split as the ids write it.
    fn short(&self) -> &'static str {
        match self.split() {
            Some("training") => "train",
            Some("test") => "test",
            _ => "val",
        }
    }
}

/// The guard of the test split, the one of `scripts/secrets2/common.py`: `test_open` exists and
/// `model` is a GGUF whose metadata holds a threshold in (0, 1) under [`TAU_KEY`]. Returns the
/// threshold.
pub fn test_is_open(model: Option<&Path>, test_open: &Path) -> Result<f64> {
    if !test_open.is_file() {
        bail!(
            "refused: the test split is closed: {} does not exist (its owner creates it by hand after training)",
            test_open.display()
        );
    }
    let Some(model) = model.filter(|m| m.is_file()) else {
        bail!("refused: the test split needs a model file whose metadata holds {TAU_KEY}");
    };
    let bytes = std::fs::read(model).with_context(|| format!("read {}", model.display()))?;
    let g = sky_gguf::Gguf::parse(&bytes)
        .map_err(|e| anyhow::anyhow!("refused: {}: not a GGUF file ({e})", model.display()))?;
    match g.get_f64(TAU_KEY) {
        Some(tau) if tau > 0.0 && tau < 1.0 => Ok(tau),
        _ => bail!(
            "refused: {}: no threshold in its metadata ({TAU_KEY}); tau* is stored before any test file is read",
            model.display()
        ),
    }
}

pub struct StatesOptions {
    /// The data folder.
    pub data: PathBuf,
    pub corpus: Corpus,
    /// The generator's raw export (`creddata:*`, `clean`, `issues:*:e2e`).
    pub candidates: Option<PathBuf>,
    /// Prefix of the outputs: `<out>.jsonl` and `<out>.records.jsonl`.
    pub out: PathBuf,
    pub tokenizer: Tokenizer,
    pub words: WordSet,
    /// The trained model, for the guard of the test split.
    pub model: Option<PathBuf>,
    /// The file that opens the test split ([`TEST_OPEN`] below the checkout).
    pub test_open: PathBuf,
}

/// What is known of one candidate besides its record.
struct Known {
    id: String,
    /// The name the record sources start with (`creddata`, `issues_e2e`…).
    name: &'static str,
    label: Option<Label>,
    /// `U` or `D` for a CredData candidate no row labels.
    place: Option<&'static str>,
    class: Option<String>,
    via: Option<&'static str>,
    /// Whether the kind question is asked (CredData, label known).
    kind: bool,
}

/// The two files being written, and the counts.
struct Sink<'a> {
    table: BufWriter<std::fs::File>,
    records: BufWriter<std::fs::File>,
    tok: &'a Tokenizer,
    words: &'a WordSet,
    source: &'static str,
    short: &'static str,
    states: usize,
    n_records: usize,
    labels: Counts,
}

impl Sink<'_> {
    fn push(&mut self, k: Known, rec: &CandRecord, scope: &[&str], ext: &str) -> Result<()> {
        let (state, f): (String, Features) = record_state(rec, scope, ext, self.tok, self.words);
        let mut row = json!({"id": k.id, "source": self.source, "split": self.short});
        if let Some(l) = k.label {
            row["label"] = json!(l.as_str());
            row["real"] = json!((l == Label::T) as u8);
        }
        if let Some(p) = k.place {
            row["unlabelled"] = json!(p);
        }
        row["class"] = json!(k.class);
        row["via"] = json!(k.via);
        row["rule"] = json!(rec.rule);
        row["features"] = features_json(&f);
        row["state"] = json!(state);
        serde_json::to_writer(&mut self.table, &row)?;
        self.table.write_all(b"\n")?;
        // the real question; its target is a placeholder when the label is not known
        let mut real = Question::noul(Q_REAL).record(
            &format!("{}:real", k.name),
            state.as_str(),
            one_hot(2, (k.label == Some(Label::T)) as usize),
        );
        real.split = "val".into();
        serde_json::to_writer(&mut self.records, &real)?;
        self.records.write_all(b"\n")?;
        self.n_records += 1;
        if let Some(l) = k.label.filter(|_| k.kind) {
            let kinds = KINDS
                .iter()
                .map(|k| (k.to_string(), String::new()))
                .collect();
            let mut kind = Question::choice(Q_KIND, kinds).record(
                &format!("{}:kind", k.name),
                state.as_str(),
                one_hot(KINDS.len(), l.kind_index()),
            );
            kind.split = "val".into();
            serde_json::to_writer(&mut self.records, &kind)?;
            self.records.write_all(b"\n")?;
            self.n_records += 1;
        }
        self.states += 1;
        bump(
            &mut self.labels,
            k.label.map(Label::as_str).or(k.place).unwrap_or("none"),
        );
        Ok(())
    }
}

/// The records of a generator export, one per line, with their index.
fn export(path: &Path) -> Result<impl Iterator<Item = Result<(usize, CandRecord)>>> {
    let f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let shown = path.display().to_string();
    Ok(BufReader::new(f)
        .lines()
        .filter(|l| !l.as_ref().is_ok_and(|l| l.trim().is_empty()))
        .enumerate()
        .map(move |(i, line)| {
            let rec: CandRecord =
                serde_json::from_str(&line?).with_context(|| format!("{shown}: candidate {i}"))?;
            if rec.generator != super::cred::GENERATOR || rec.masked {
                bail!("{shown}: candidate {i} is not a raw record of the generator");
            }
            Ok((i, rec))
        }))
}

/// Builds `<out>.jsonl` and `<out>.records.jsonl`; returns the counts and the hashes.
pub fn build(o: &StatesOptions) -> Result<serde_json::Value> {
    let test = o.corpus.split() == Some("test");
    if test {
        test_is_open(o.model.as_deref(), &o.test_open)?;
    }
    if let Some(dir) = o.out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let with = |ext: &str| PathBuf::from(format!("{}.{ext}", o.out.display()));
    let (table_path, records_path) = (with("jsonl"), with("records.jsonl"));
    let create = |p: &Path| -> Result<BufWriter<std::fs::File>> {
        Ok(BufWriter::new(
            std::fs::File::create(p).with_context(|| format!("create {}", p.display()))?,
        ))
    };
    let mut sink = Sink {
        table: create(&table_path)?,
        records: create(&records_path)?,
        tok: &o.tokenizer,
        words: &o.words,
        source: match o.corpus {
            Corpus::CredData { .. } => super::CREDDATA,
            Corpus::Clean => "clean",
            Corpus::Issues { .. } => super::ISSUES,
        },
        short: o.corpus.short(),
        states: 0,
        n_records: 0,
        labels: Counts::new(),
    };
    let need_export = || {
        o.candidates
            .as_deref()
            .context("--candidates (the generator's raw export) is required for this corpus")
    };
    let mut counts = Counts::new();
    let input = match &o.corpus {
        Corpus::CredData { split } => {
            let path = need_export()?;
            creddata(o, split, test, path, &mut sink, &mut counts)?;
            file_entry(path)?
        }
        Corpus::Clean => {
            let path = need_export()?;
            for item in export(path)? {
                let (index, rec) = item?;
                counts.insert("candidates".into(), index + 1);
                // the path inside its repository: the first component names the repository
                let file = rec.file.as_deref().context("a candidate without a file")?;
                let inner = file.split_once('/').map_or(file, |(_, rest)| rest);
                let (scope, ext) = sc::path_scope(inner);
                let known = unlabelled(format!("clean:all:{index}"), "clean");
                sink.push(known, &rec, &scope, &ext)?;
            }
            file_entry(path)?
        }
        Corpus::Issues { split, variant } => {
            let base = o.data.join("eval/issues").join(split);
            let index = read_json(&base.join("index.json")).with_context(|| {
                format!(
                    "{}: scripts/secrets2/issues.py prepare writes it",
                    base.display()
                )
            })?;
            if index["split"] != split.as_str() {
                bail!(
                    "{}: not the issue texts of the {split} split",
                    base.display()
                );
            }
            match variant {
                Variant::EndToEnd => {
                    let path = need_export()?;
                    for item in export(path)? {
                        let (i, rec) = item?;
                        counts.insert("candidates".into(), i + 1);
                        // an issue text has no path: no scope, no extension
                        if rec.file.is_some() {
                            bail!("{}: candidate {i} has a path; the issue texts are read without one", path.display());
                        }
                        let id = format!("issue-e2e:{}:{i}", o.corpus.short());
                        sink.push(unlabelled(id, "issues_e2e"), &rec, &[], "")?;
                    }
                    file_entry(path)?
                }
                _ => {
                    let obfuscated = *variant == Variant::Obfuscated;
                    issue_rows(&base, obfuscated, o.corpus.short(), &mut sink, &mut counts)?;
                    file_entry(&base.join("candidates.jsonl"))?
                }
            }
        }
    };
    sink.table.flush()?;
    sink.records.flush()?;
    let (states, n_records, labels) = (sink.states, sink.n_records, sink.labels.clone());
    drop(sink);
    let candidates = counts.remove("candidates").unwrap_or(0);
    Ok(json!({
        "corpus": corpus_name(&o.corpus),
        "candidates": candidates,
        "states": states,
        "without_a_scope_directory": counts.remove("without_a_scope_directory").unwrap_or(0),
        "not_found_in_their_text": counts.remove("not_found").unwrap_or(0),
        "records": n_records,
        "labels": labels,
        "input": input,
        "table": file_entry(&table_path)?,
        "records_file": file_entry(&records_path)?,
    }))
}

fn corpus_name(c: &Corpus) -> String {
    match c {
        Corpus::Clean => "clean".into(),
        Corpus::CredData { split } => format!("creddata:{split}"),
        Corpus::Issues { split, variant } => match variant {
            Variant::Published => format!("issues:{split}"),
            Variant::Obfuscated => format!("issues:{split}:obfuscated"),
            Variant::EndToEnd => format!("issues:{split}:e2e"),
        },
    }
}

fn unlabelled(id: String, name: &'static str) -> Known {
    Known {
        id,
        name,
        label: None,
        place: None,
        class: None,
        via: None,
        kind: false,
    }
}

/// The CredData candidates of a split: every record of the export, labelled from the meta rows
/// of its file as the training build labels (T before X before F; `U` on a line no row covers,
/// `D` on a labelled line over no labelled value).
fn creddata(
    o: &StatesOptions,
    split: &str,
    test_open: bool,
    path: &Path,
    sink: &mut Sink<'_>,
    counts: &mut Counts,
) -> Result<()> {
    let splits = Splits::read(&o.data)?;
    let repos: HashSet<&str> = match split {
        "training" => &splits.training,
        "validation" => &splits.validation,
        _ => &splits.test,
    }
    .iter()
    .map(String::as_str)
    .collect();
    let cred = o.data.join("creddata");
    let mut meta: HashMap<String, Vec<MetaRow>> = HashMap::new();
    // the rows of the file being read, placed in its text
    let mut current: Option<(String, FileRows)> = None;
    let short = sink.short;
    for item in export(path)? {
        let (index, rec) = item?;
        counts.insert("candidates".into(), index + 1);
        let file = rec.file.clone().context("a candidate without a file")?;
        let repo = repo_of(&file)?.to_string();
        if !repos.contains(repo.as_str()) {
            bail!("refused: {file} is not a file of the {split} split");
        }
        if file.split('/').count() == 3 {
            // a licence copy at the repository's root: no scope directory, no state
            bump(counts, "without_a_scope_directory");
            continue;
        }
        if current.as_ref().map(|c| c.0.as_str()) != Some(file.as_str()) {
            if !meta.contains_key(&repo) {
                meta.insert(
                    repo.clone(),
                    read_meta_eval(&cred, &splits, &repo, test_open)?,
                );
            }
            let rows: Vec<&MetaRow> = meta[&repo].iter().filter(|r| r.file == file).collect();
            let u = read_utf16(&cred.join(&file))?
                .with_context(|| format!("{file}: the generator read it, it is empty or binary"))?;
            current = Some((file.clone(), FileRows::new(&u, &rows)));
        }
        let placed = &current.as_ref().expect("the file just placed").1;
        let id = format!("cred:{short}:{index}");
        let known = match placed.label(rec.offset, rec.offset + rec.length) {
            Own::Row {
                label,
                via,
                category,
                ..
            } => Known {
                id,
                name: "creddata",
                label: Some(label),
                place: None,
                class: Some(category),
                via: Some(via),
                kind: true,
            },
            own => Known {
                place: Some(if own == Own::Dropped { "D" } else { "U" }),
                ..unlabelled(id, "creddata_unlabelled")
            },
        };
        sink.push(known, &rec, &stored_scope(&file)?, &rec.ext)?;
    }
    Ok(())
}

/// The dataset's candidates of an issue-report split, from what `issues.py prepare` wrote:
/// `candidates.jsonl` (row, text id, label, place in code points) and the texts as files.
fn issue_rows(
    base: &Path,
    obfuscated: bool,
    short: &str,
    sink: &mut Sink<'_>,
    counts: &mut Counts,
) -> Result<()> {
    let texts = base.join(if obfuscated {
        "texts-obfuscated"
    } else {
        "texts"
    });
    let name = if obfuscated {
        "issues_obfuscated"
    } else {
        "issues"
    };
    let list = base.join("candidates.jsonl");
    let f = std::fs::File::open(&list).with_context(|| format!("open {}", list.display()))?;
    let mut cache: Option<(String, String)> = None;
    for (n, line) in BufReader::new(f).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let c: serde_json::Value =
            serde_json::from_str(&line).with_context(|| format!("{}:{}", list.display(), n + 1))?;
        *counts.entry("candidates".into()).or_default() += 1;
        let (Some(row), Some(tid)) = (c["row"].as_u64(), c["text"].as_str()) else {
            bail!(
                "{}:{}: not a row of candidates.jsonl",
                list.display(),
                n + 1
            );
        };
        let (Some(start), Some(end)) = (c["start"].as_u64(), c["end"].as_u64()) else {
            bump(counts, "not_found");
            continue;
        };
        if tid.len() != 16 || !tid.bytes().all(|b| b.is_ascii_hexdigit()) {
            bail!(
                "{}:{}: {tid:?} is not the id of an issue text",
                list.display(),
                n + 1
            );
        }
        if cache.as_ref().map(|t| t.0.as_str()) != Some(tid) {
            let path = texts.join(format!("{tid}.txt"));
            let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
            cache = Some((
                tid.to_string(),
                String::from_utf8_lossy(&bytes).into_owned(),
            ));
        }
        let text = &cache.as_ref().expect("the text just read").1;
        // the place is in code points (Python indices)
        let at = |cp: u64| {
            text.char_indices()
                .nth(cp as usize)
                .map_or(text.len(), |(i, _)| i)
        };
        let (a, b) = (at(start), at(end));
        if b <= a {
            bail!("{}:{}: an empty candidate", list.display(), n + 1);
        }
        let rec = record_at(
            issues::GENERATOR,
            None,
            issues::RULE,
            None,
            text,
            a,
            b,
            RecFilters::default(),
        );
        let label = if c["label"].as_u64() == Some(1) {
            Label::T
        } else {
            Label::F
        };
        let known = Known {
            label: Some(label),
            ..unlabelled(format!("issue:{short}:{row}"), name)
        };
        sink.push(known, &rec, &[], "")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpora_are_named_as_the_evaluation_names_them() {
        assert_eq!(Corpus::parse("clean").unwrap(), Corpus::Clean);
        let c = Corpus::parse("creddata:validation").unwrap();
        assert_eq!((c.split(), c.short()), (Some("validation"), "val"));
        let c = Corpus::parse("issues:test:obfuscated").unwrap();
        assert_eq!((c.split(), c.short()), (Some("test"), "test"));
        assert_eq!(corpus_name(&c), "issues:test:obfuscated");
        assert_eq!(
            Corpus::parse("issues:training:e2e").unwrap().short(),
            "train"
        );
        for bad in [
            "creddata",
            "creddata:val",
            "issues:validation:other",
            "clean:all",
            "",
        ] {
            assert!(Corpus::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_test_split_needs_the_open_file_and_a_model_with_its_threshold() {
        let dir = std::env::temp_dir().join(format!("sky-states-guard-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (open, model) = (dir.join("TEST-OPEN"), dir.join("m.gguf"));
        let gguf = |tau: Option<f32>| {
            let mut w = sky_gguf::GgufWriter::new();
            w.add_kv("general.name", sky_gguf::OwnedValue::String("m".into()));
            if let Some(t) = tau {
                w.add_kv(TAU_KEY, t);
            }
            std::fs::write(&model, w.to_bytes().unwrap()).unwrap();
        };
        gguf(Some(0.4));
        // closed: the file is not there, whatever the model
        let err = test_is_open(Some(&model), &open).unwrap_err().to_string();
        assert!(err.contains("the test split is closed"), "{err}");
        std::fs::write(&open, "").unwrap();
        // open, but no model, a file that is no GGUF, a model without its threshold
        assert!(test_is_open(None, &open).is_err());
        assert!(test_is_open(Some(&dir.join("missing.gguf")), &open).is_err());
        std::fs::write(dir.join("x.gguf"), "not a model").unwrap();
        assert!(test_is_open(Some(&dir.join("x.gguf")), &open).is_err());
        gguf(None);
        let err = test_is_open(Some(&model), &open).unwrap_err().to_string();
        assert!(err.contains(TAU_KEY), "{err}");
        for tau in [0.0, 1.0] {
            gguf(Some(tau));
            assert!(test_is_open(Some(&model), &open).is_err(), "{tau}");
        }
        gguf(Some(0.4));
        assert!((test_is_open(Some(&model), &open).unwrap() - 0.4).abs() < 1e-6);
        std::fs::remove_dir_all(dir).ok();
    }
}
