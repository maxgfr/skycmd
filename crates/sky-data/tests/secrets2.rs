//! The s6-secrets v2 data build on a miniature data folder (`docs/s6-secrets2-prereg.md`,
//! section 3): three repositories of three lines' worth of CredData, three issue-report splits
//! of a few rows, written here with invented values. The build runs from the "frozen" generator
//! outputs to `BUILD.json`, twice, and the test checks what the pre-registration asks of it:
//! labels, decontamination, weak negatives, replacement, the files of the mix, and that the same
//! inputs give the same bytes. The test repository has no meta file at all: a build that opened
//! one would fail.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use sky_data::record::{read_jsonl, validate_record, DecisionRecord};
use sky_data::secrets2::{self as s2, cred, issues, mix, synth, Item, Label, RecFilters};

const TRAIN: &str = "aaaaaaaa";
/// A second training repository, before the first in the order of the RepoIDs.
const SECOND: &str = "a0a0a0a0";
const VAL: &str = "bbbbbbbb";
const TEST: &str = "cccccccc";

/// Invented values: letters and digits of no provider's format.
fn value(tag: &str, n: usize) -> String {
    let set: Vec<char> = "QwErTyUiOpAsDfGhJkLzXcVbNm0123456789".chars().collect();
    let mut h = sky_data::rng::hash64(tag.as_bytes());
    (0..n)
        .map(|_| {
            h = sky_data::rng::mix64(h.wrapping_add(0x9E37_79B9_7F4A_7C15));
            set[(h % set.len() as u64) as usize]
        })
        .collect()
}

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

const META_HEADER: &str = "Id,FileID,Domain,RepoName,FilePath,LineStart,LineEnd,GroundTruth,ValueStart,ValueEnd,CryptographyKey,PredefinedPattern,Category";

/// One line of a labelled file: its text, the value a candidate covers, the meta row on it
/// (`T`, `X`, `F` with the value's span, `L` for a whole-line F row, `O` for a row over
/// something else on the line) and the label the frozen file gives the candidate.
struct Line {
    text: String,
    value: String,
    row: Option<char>,
    label: &'static str,
}

fn line(name: &str, value: &str, row: Option<char>, label: &'static str) -> Line {
    Line {
        text: format!("{name} = \"{value}\" # note"),
        value: value.to_string(),
        row,
        label,
    }
}

/// Writes a repository's file, its meta rows, and returns its candidates with their frozen
/// labels (as lines of the two generator files).
fn repository(root: &Path, repo: &str, file_id: &str, lines: &[Line]) -> (Vec<Value>, Vec<Value>) {
    let file = format!("data/{repo}/src/conf/{file_id}.py");
    let text: String = lines.iter().map(|l| format!("{}\n", l.text)).collect();
    write(&root.join("creddata").join(&file), &text);
    let mut meta = format!("{META_HEADER}\n");
    let (mut cands, mut labels) = (Vec::new(), Vec::new());
    let mut offset = 0;
    for (i, l) in lines.iter().enumerate() {
        let col = l.text.find(&l.value).unwrap();
        let id = 100 + i;
        let n = i + 1;
        let row = |truth: &str, a: String, b: String, cat: &str| {
            format!("{id},{file_id},GitHub,{repo},{file},{n},{n},{truth},{a},{b},,,{cat}\n")
        };
        let span = (col.to_string(), (col + l.value.len()).to_string());
        match l.row {
            Some('T') => meta += &row("T", span.0, span.1, "Password"),
            Some('X') => meta += &row("X", span.0, span.1, "Token"),
            Some('F') => meta += &row("F", span.0, span.1, "Key"),
            Some('L') => meta += &row("F", String::new(), String::new(), "Secret"),
            Some('O') => {
                meta += &row(
                    "F",
                    (l.text.len() - 4).to_string(),
                    l.text.len().to_string(),
                    "Key",
                )
            }
            _ => {}
        }
        let at = offset + col;
        let mut rec = s2::record_at(
            cred::GENERATOR,
            Some(&file),
            "generic-keyword",
            l.text.split(' ').next(),
            &text,
            at,
            at + l.value.len(),
            RecFilters::default(),
        );
        rec.revision = Some(cred::REVISION);
        let mut label = json!({
            "index": cands.len(), "file": file, "offset": rec.offset, "length": rec.length,
            "rule": rec.rule, "split": "x", "repo": repo, "label": l.label,
        });
        if let Some(r) = l.row.filter(|_| matches!(l.label, "T" | "X" | "F")) {
            label["via"] = json!(if r == 'L' { "line" } else { "value" });
            label["rowId"] = json!(id.to_string());
            label["category"] = json!(match r {
                'T' => "Password",
                'X' => "Token",
                'L' => "Secret",
                _ => "Key",
            });
            label["rows"] = json!(1);
        }
        cands.push(serde_json::to_value(&rec).unwrap());
        labels.push(label);
        offset += l.text.len() + 1;
    }
    write(&root.join(format!("creddata/meta/{repo}.csv")), &meta);
    (cands, labels)
}

fn jsonl(path: &Path, rows: &[Value]) {
    let text: String = rows.iter().map(|r| format!("{r}\n")).collect();
    write(path, &text);
}

/// A CSV field, quoted.
fn q(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}

struct Fixture {
    root: PathBuf,
    /// Values that must be in no output: the issue-report positives as published.
    published: Vec<String>,
    test_csv: String,
}

/// The miniature data folder.
fn fixture(root: &Path) -> Fixture {
    std::fs::remove_dir_all(root).ok();
    let v = |tag: &str| value(tag, 20);
    // the training repository: what each candidate must become
    let train = [
        line("db_password", &v("t1"), Some('T'), "T"), // its shape is in the test repository: removed
        line("service_secret", &v("t2"), Some('T'), "T"), // kept
        line("api_token", &v("f1"), Some('L'), "F"), // its line is in the test repository: removed
        line("session_key", &v("f2"), Some('F'), "F"), // kept
        line("user_secret", &v("x1"), Some('X'), "X"), // kept
        line("auth_note", &v("d1"), Some('O'), "D"), // on a labelled line, over no value: dropped
        line("cache_key", &v("u1"), None, "U"), // its line is in the test repository: out of the pool
        line("debug_key", &v("u2"), None, "U"),
        line("trace_token", &v("u3"), None, "U"),
        line("build_key", &v("u4"), None, "U"),
        line("admin_pass", &v("f3"), Some('F'), "F"), // kept
        // one line six times: the four in the middle have the same state
        line("flag_token", "false-flag", None, "U"),
        line("flag_token", "false-flag", None, "U"),
        line("flag_token", "false-flag", None, "U"),
        line("flag_token", "false-flag", None, "U"),
        line("flag_token", "false-flag", None, "U"),
        line("flag_token", "false-flag", None, "U"),
    ];
    // a second training repository: eight labelled candidates and two on unlabelled lines
    let mut second: Vec<Line> = (0..8)
        .map(|i| {
            line(
                &format!("option_key_{i}"),
                &v(&format!("s{i}")),
                Some('F'),
                "F",
            )
        })
        .collect();
    second.push(line("region_key", &v("su1"), None, "U"));
    second.push(line("zone_token", &v("su2"), None, "U"));
    let val = [
        line("root_password", &v("vt"), Some('T'), "T"),
        line("page_token", &v("vf"), Some('F'), "F"),
        line("tmp_key", &v("vu"), None, "U"),
    ];
    // the candidates file: the repositories in the order of their RepoID
    let (mut tc, mut tl) = repository(root, SECOND, "0d0d0d0d", &second);
    let (c, l) = repository(root, TRAIN, "0a0a0a0a", &train);
    tc.extend(c);
    tl.extend(l);
    for (i, l) in tl.iter_mut().enumerate() {
        l["index"] = json!(i);
    }
    let (vc, vl) = repository(root, VAL, "0b0b0b0b", &val);
    // the test repository: lines only, no meta file
    let test_lines = [
        format!("  db_password =   \"{}\" # note  ", v("other-value")),
        train[2].text.clone(),
        format!("\t{}", train[6].text),
        "something else entirely".to_string(),
    ];
    write(
        &root.join(format!("creddata/data/{TEST}/_/0c0c0c0c.txt")),
        &test_lines.join("\r\n"),
    );
    write(
        &root.join(format!("creddata/data/{TEST}/LICENSE")),
        "Permission is hereby granted\n",
    );
    write(
        &root.join("creddata/word_in_path.json"),
        &json!(sky_infer::secret_cand::WORD_IN_PATH.as_slice()).to_string(),
    );
    write(
        &root.join("splits.json"),
        &json!({
            "training": [{"repo_id": TRAIN}, {"repo_id": SECOND}],
            "validation": [{"repo_id": VAL}],
            "test": [{"repo_id": TEST}],
        })
        .to_string(),
    );
    let gen = root.join("gen");
    jsonl(&gen.join("candidates-training.raw.jsonl"), &tc);
    jsonl(&gen.join("labels-training.jsonl"), &tl);
    jsonl(&gen.join("candidates-validation.raw.jsonl"), &vc);
    jsonl(&gen.join("labels-validation.jsonl"), &vl);
    let mut files = serde_json::Map::new();
    for name in [
        "candidates-training.raw.jsonl",
        "labels-training.jsonl",
        "candidates-validation.raw.jsonl",
        "labels-validation.jsonl",
    ] {
        files.insert(
            name.into(),
            json!({"sha256": s2::sha256_file(&gen.join(name)).unwrap()}),
        );
    }
    write(
        &gen.join("generator.json"),
        &json!({"generator": "sg-cand-1", "revision": 1, "frozen": "t", "secretgate": {"commit": "c"}, "files": files}).to_string(),
    );

    // the issue-report splits
    let (p1, p2, p3) = (
        format!("Winter-{}", value("p1", 14)),
        format!("api_key={}", value("p2", 32)),
        format!("token: {}", value("p3", 24)),
    );
    let shared = "a text that is also in the test split";
    let header = ",Unnamed: 0,candidate_string,text,Issue_id,label\n";
    let row = |i: usize, c: &str, t: &str, l: u8| format!("{i},{i},{},{},u{i},{l}\n", q(c), q(t));
    let t1 = format!("It fails.\r\nconfig:\r\n  password = {p1}\r\nThanks");
    let t2 = format!("Same key as in the other issue: {p1} (please rotate)\nbuild 4f2a9c\n");
    let t3 = format!("log line\nkey={p2}\nend");
    // five distinct negatives hold `token`, `api` and `key` as whole runs: a context word
    // (amendment 4) and a key name that is no dictionary word (amendment 5)
    let context: String = (5..10)
        .map(|i| {
            let c = format!("api_key={i}x token: n{i}");
            row(i, &c, &format!("an issue\n  {c}\nno secret here"), 0)
        })
        .collect();
    let train_csv = format!(
        "{header}{}{}{}{}{}{context}",
        row(0, &p1, &t1, 1),
        row(1, "4f2a9c", &t2, 0),
        row(2, &p2, &t3, 1),
        row(3, "also in the test", shared, 0),
        row(4, "nowhere", "a text without its candidate", 0),
    );
    let v1 = format!("see\n{p3}\nabove");
    let val_csv = format!("{header}{}{}", row(0, &p3, &v1, 1), row(1, "above", &v1, 0));
    let test_secret = value("test-secret", 30);
    let test_csv = format!(
        "{header}{}{}",
        row(0, &test_secret, &format!("real one: {test_secret}"), 1),
        row(1, "is also", shared, 0),
    );
    let dir = root.join(issues::DATA_DIR);
    write(&dir.join("train.csv"), &train_csv);
    write(&dir.join("val.csv"), &val_csv);
    write(&dir.join("test.csv"), &test_csv);
    write(
        &root.join("words.txt"),
        "winter\nsummer\nspring\nautumn\ntree\nbird\nlamp\nhouse\nhorse\nmouse\ntoken\nmaple\nNotThis\nab\n",
    );
    Fixture {
        root: root.to_path_buf(),
        published: vec![p1, p2, p3],
        test_csv,
    }
}

fn tokenizer() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../sky-infer/tests/fixtures/small_tokenizer_s2.json")
}

/// The whole build into `<root>/<out>`, as `scripts/build-secrets2-data.sh` chains it.
fn build(f: &Fixture, out: &str) -> PathBuf {
    let out = f.root.join(out);
    let words = || s2::load_words(&f.root.join("words.txt")).unwrap();
    cred::build(&cred::CredOptions {
        data: f.root.clone(),
        out: out.clone(),
        seed: 1234,
        tokenizer: s2::load_tokenizer(&tokenizer()).unwrap(),
        words: words(),
    })
    .unwrap();
    issues::build(&issues::IssueOptions {
        data: f.root.clone(),
        out: out.clone(),
        seed: 1234,
        words: words(),
    })
    .unwrap();
    synth::write(&out, 1234, None, 40, &words()).unwrap();
    mix::build(&mix::MixOptions {
        out: out.clone(),
        tokenizer: s2::load_tokenizer(&tokenizer()).unwrap(),
        words: words(),
    })
    .unwrap();
    // what `sky-train s2-truncation --json` writes
    let report = json!({
        "creddata:real": {"n": 5, "state": 0.2, "head": 0.0, "any": 0.2},
        "issues:real": {"n": 4, "state": 0.0, "head": 0.0, "any": 0.0},
        "all": {"n": 9, "state": 0.1111, "head": 0.0, "any": 0.1111},
    });
    s2::write_json(&out.join("truncation.json"), &report).unwrap();
    s2::build::manifest(&s2::build::ManifestOptions {
        data: f.root.clone(),
        out: out.clone(),
        seed: 1234,
        dict: f.root.join("words.txt"),
        tokenizer: tokenizer(),
        truncation: Some(out.join("truncation.json")),
        hand_read: None,
        config: None,
    })
    .unwrap();
    out
}

fn files_under(dir: &Path, rel: &str, out: &mut BTreeMap<String, Vec<u8>>) {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap())
        .collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let name = format!("{rel}/{}", e.file_name().to_string_lossy());
        if e.file_type().unwrap().is_dir() {
            files_under(&e.path(), &name, out);
        } else {
            out.insert(name, std::fs::read(e.path()).unwrap());
        }
    }
}

fn items(out: &Path, name: &str) -> Vec<Item> {
    read_jsonl(&out.join(format!("candidates/{name}.jsonl"))).unwrap()
}

#[test]
fn the_build_does_what_the_pre_registration_asks() {
    let root = std::env::temp_dir().join(format!("sky-secrets2-{}", std::process::id()));
    let f = fixture(&root);
    let out = build(&f, "build");

    // CredData: the frozen labels and this build's own reading agree
    let c = s2::read_json(&out.join("counts/creddata.json")).unwrap();
    for split in ["training", "validation"] {
        assert_eq!(
            c["labels"][split]["counts"]["label_differences"], 0,
            "{split}"
        );
        assert_eq!(
            c["labels"][split]["counts"]["via_differences"], 0,
            "{split}"
        );
        assert_eq!(
            c["labels"][split]["counts"]["row_differences"], 0,
            "{split}"
        );
        assert_eq!(
            c["labels"][split]["frozen_labels"],
            c["labels"][split]["own_labels"]
        );
    }
    assert_eq!(
        c["labels"]["training"]["frozen_labels"],
        json!({"D": 1, "F": 11, "T": 2, "U": 12, "X": 1})
    );
    // decontamination: the T whose shape, the F whose line and the pool candidate whose line
    // are in the test repository
    let removed = &c["items"]["training"]["removed_by_decontamination"];
    assert_eq!(removed["T"], json!({"value_replaced_by_its_length": 1}));
    assert_eq!(removed["F"], json!({"exact": 1}));
    assert_eq!(c["decontamination"]["test_files_read"], 2);
    assert_eq!(c["decontamination"]["test_lines"], 4 + 2);
    // the weak negatives (amendment 4): the pool is reduced to one candidate per distinct
    // state, then the repositories give one candidate each in turn, in the order of their ids
    let w = &c["weak_negatives"];
    assert_eq!(w["pool"], 12);
    assert_eq!(w["pool_removed_by_decontamination"], 1);
    assert_eq!(w["pool_left"], 11);
    // the line written six times is three states: its first, its last, the four between
    assert_eq!(w["pool_one_per_distinct_state"], 8);
    assert_eq!(w["pool_repositories"], 2);
    // 12 labelled candidates are kept: three weak negatives at most
    assert_eq!(w["labelled_training_candidates_kept"], 12);
    assert_eq!((w["cap"].as_u64(), w["drawn"].as_u64()), (Some(3), Some(3)));
    assert_eq!(w["by_repository"], json!({SECOND: 2, TRAIN: 1}));
    assert_eq!(
        (
            w["repositories_contributing"].as_u64(),
            w["largest_contribution"].as_u64()
        ),
        (Some(2), Some(2))
    );
    let train = items(&out, "creddata-train");
    let of = |repo: &str, weak: bool| -> Vec<(&str, &str)> {
        train
            .iter()
            .filter(|i| i.weak == weak && i.rec.file.as_deref().unwrap().contains(repo))
            .map(|i| (i.label.as_str(), i.rec.name.as_deref().unwrap()))
            .collect()
    };
    assert_eq!(
        of(TRAIN, false),
        [
            ("T", "service_secret"),
            ("F", "session_key"),
            ("X", "user_secret"),
            ("F", "admin_pass"),
        ]
    );
    assert_eq!(of(SECOND, false).len(), 8);
    assert_eq!(train.len(), 12 + 3);
    assert_eq!(of(SECOND, true), [("F", "region_key"), ("F", "zone_token")]);
    let weak = of(TRAIN, true);
    assert_eq!(weak.len(), 1);
    assert!(
        weak[0].0 == "F"
            && ["debug_key", "trace_token", "build_key", "flag_token"].contains(&weak[0].1)
    );
    assert!(train
        .iter()
        .all(|i| i.source == "creddata" && i.split == "train" && i.kind));
    // an id is the index of the candidate in the candidates file
    let secret = train
        .iter()
        .find(|i| i.rec.name.as_deref() == Some("service_secret"))
        .unwrap();
    assert_eq!(secret.id, "cred:train:11");
    let val = items(&out, "creddata-val");
    assert_eq!(
        val.iter().map(|i| i.label).collect::<Vec<_>>(),
        [Label::T, Label::F]
    );

    // issue reports: the shared text is removed, the positives replaced wherever they stand
    let i = s2::read_json(&out.join("counts/issues.json")).unwrap();
    assert_eq!(i["splits"]["train"]["removed_text_in_test"], 1);
    assert_eq!(i["splits"]["train"]["not_found"], 1);
    assert_eq!(i["splits"]["train"]["candidates"], 8);
    let r = &i["replacement"];
    assert_eq!(r["published_positive_strings_left_in_a_record"], 0);
    assert_eq!(r["unchanged_made_of_kept_words_and_separators"], 0);
    // the word rule (amendments 4 and 5): `token`, `api` and `key` stand in five distinct
    // negatives of the training split and are kept in the positives that hold them; `Winter`
    // is still replaced
    assert_eq!(r["kept_runs_of_the_training_negatives"], 3);
    assert_eq!(r["positive_strings_with_a_kept_run"], 2);
    assert_eq!(r["positive_strings_with_a_replaced_word_run"], 1);
    assert_eq!(r["kept_runs_in_the_positive_strings"], 3);
    assert_eq!(
        r["ten_most_frequent_kept_runs"],
        json!([
            {"run": "api", "positive_strings": 1},
            {"run": "key", "positive_strings": 1},
            {"run": "token", "positive_strings": 1},
        ])
    );
    let key = items(&out, "issues-train");
    assert!(
        key[2].rec.value.starts_with("api_key="),
        "the key name stays"
    );
    assert_ne!(key[2].rec.value, f.published[1]);
    let v = items(&out, "issues-val");
    assert_eq!(v[0].label, Label::T);
    assert!(
        v[0].rec.value.starts_with("token: "),
        "the context word stays"
    );
    assert_ne!(v[0].rec.value, f.published[2]);
    assert_eq!(v[0].rec.value.len(), f.published[2].len());
    let it = items(&out, "issues-train");
    assert_eq!(
        it.iter().map(|i| i.label).collect::<Vec<_>>()[..3],
        [Label::T, Label::F, Label::T]
    );
    assert_eq!(it.len(), 8);
    assert!(it
        .iter()
        .all(|i| !i.kind && i.rec.file.is_none() && i.rec.rule == issues::RULE));
    // the first positive keeps its word as a word and its shape
    let new = &it[0].rec.value;
    assert_eq!(new.len(), f.published[0].len());
    assert!(new[..6].chars().all(|c| c.is_ascii_alphabetic()) && &new[6..7] == "-");
    assert!(
        ["Summer", "Spring", "Autumn"].contains(&&new[..6]),
        "a word of the dictionary"
    );
    assert_eq!(it[0].rec.line_text, format!("  password = {new}"));
    assert_eq!(it[0].rec.prev_line.as_deref(), Some("config:"));
    // the negative of another issue that quotes the same value sees its replacement
    assert!(it[1]
        .rec
        .prev_line
        .as_deref()
        .unwrap()
        .contains(new.as_str()));
    assert_eq!(it[1].rec.value, "4f2a9c");

    // nothing the build wrote holds a positive as published, and the test split is untouched
    let mut all = BTreeMap::new();
    files_under(&out, "", &mut all);
    for (name, bytes) in &all {
        let text = String::from_utf8_lossy(bytes);
        for p in &f.published {
            assert!(
                !text.contains(p.as_str()),
                "{name} holds a published positive"
            );
        }
        assert!(
            !text.contains(&value("test-secret", 30)),
            "{name} holds a value of the test split"
        );
    }
    assert_eq!(
        std::fs::read_to_string(root.join(issues::DATA_DIR).join("test.csv")).unwrap(),
        f.test_csv
    );

    // the mix: one directory per source, questions as section 1 words them
    for source in mix::SOURCES {
        for split in mix::SPLITS {
            let recs: Vec<DecisionRecord> =
                read_jsonl(&out.join(format!("records/{source}/{split}.jsonl"))).unwrap();
            assert!(!recs.is_empty(), "{source} {split}");
            for r in &recs {
                validate_record(r).unwrap();
                assert_eq!(r.split, split);
                assert!(r.src.starts_with(source), "{}", r.src);
                assert!(r.state.starts_with("rule ") && r.state.contains("\nlen "));
            }
            let kinds = recs.iter().filter(|r| r.t == "choice").count();
            assert_eq!(
                kinds,
                if source == "issues" {
                    0
                } else {
                    recs.len() / 2
                }
            );
        }
    }
    let recs: Vec<DecisionRecord> = read_jsonl(&out.join("records/creddata/train.jsonl")).unwrap();
    assert_eq!(
        recs.iter()
            .filter(|r| r.src == "creddata_weak:real")
            .count(),
        3
    );
    let secret = recs
        .iter()
        .find(|r| r.state.contains("\nkey service_secret\n") && r.t == "noul")
        .unwrap();
    assert!(secret
        .state
        .starts_with("rule generic-keyword\npath src conf .py\nkey service_secret\nvalue "));
    assert_eq!(secret.target, [0.0, 1.0]);
    let issue: Vec<DecisionRecord> = read_jsonl(&out.join("records/issues/train.jsonl")).unwrap();
    assert!(issue[0]
        .state
        .starts_with("rule issue-regex\npath _\nkey\nvalue "));
    // synthetic candidates: a third of the two other sources at most
    let m = s2::read_json(&out.join("counts/mix.json")).unwrap();
    assert_eq!(
        m["training_candidates"],
        json!({"creddata": 15, "issues": 8, "synth": 7, "total": 30, "synthetic_share": 0.2333})
    );
    assert_eq!(m["weights"], mix::WEIGHTS);

    // BUILD.json: the window rule on one state per candidate, and every output with its hash
    let b = s2::read_json(&out.join("BUILD.json")).unwrap();
    assert_eq!(b["truncation"]["states"], 9);
    assert_eq!(b["truncation"]["truncated_at_320"], 1);
    assert_eq!(b["truncation"]["window"], 384);
    assert_eq!(b["outputs"].as_object().unwrap().len(), 18 + 1);
    assert_eq!(
        b["outputs"]["records/synth/val.jsonl"]["sha256"],
        s2::sha256_file(&out.join("records/synth/val.jsonl")).unwrap()
    );

    // the same inputs and the same seed: the same bytes
    let again = build(&f, "again");
    let mut second = BTreeMap::new();
    files_under(&again, "", &mut second);
    assert_eq!(
        all.keys().collect::<Vec<_>>(),
        second.keys().collect::<Vec<_>>()
    );
    for (name, bytes) in &all {
        assert!(bytes == &second[name], "{name} differs between two builds");
    }
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn the_build_refuses_what_is_not_the_frozen_generator() {
    let root = std::env::temp_dir().join(format!("sky-secrets2-frozen-{}", std::process::id()));
    let f = fixture(&root);
    let run = || {
        cred::build(&cred::CredOptions {
            data: f.root.clone(),
            out: f.root.join("build"),
            seed: 1234,
            tokenizer: s2::load_tokenizer(&tokenizer()).unwrap(),
            words: s2::load_words(&f.root.join("words.txt")).unwrap(),
        })
    };
    // a candidates file that is not the one generator.json names
    let cands = root.join("gen/candidates-validation.raw.jsonl");
    let text = std::fs::read_to_string(&cands).unwrap();
    std::fs::write(&cands, format!("{text}\n")).unwrap();
    let err = run().unwrap_err().to_string();
    assert!(err.contains("sha256"), "{err}");
    std::fs::write(&cands, text).unwrap();
    run().unwrap();
    // the candidates of a repository that is a test repository: refused by the guard, before
    // any meta file is opened
    let splits = json!({
        "training": [{"repo_id": TRAIN}, {"repo_id": SECOND}],
        "validation": [{"repo_id": TEST}],
        "test": [{"repo_id": VAL}],
    });
    std::fs::write(root.join("splits.json"), splits.to_string()).unwrap();
    let err = run().unwrap_err().to_string();
    assert_eq!(err, format!("refused: {VAL} is a test repository"));
    // no generator.json: nothing is built
    std::fs::remove_file(root.join("gen/generator.json")).unwrap();
    let err = format!("{:#}", run().unwrap_err());
    assert!(err.contains("not frozen"), "{err}");
    std::fs::remove_dir_all(&root).ok();
}
