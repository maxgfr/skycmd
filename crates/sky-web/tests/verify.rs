//! The secrets verifier ([`sky_web::verify`]) on the served target. Runs natively
//! (`cargo test -p sky-web --test verify`) and on `wasm32-unknown-unknown`
//! (`wasm-pack test --node crates/sky-web --test verify`):
//!
//! - **states**: the candidate JSON gives the states of `sky-infer/tests/fixtures/secret_cand.json`
//!   (with a real path, and with the scope and extension given instead);
//! - **the JSON contract**: missing optional fields have their defaults, a wrong type is refused
//!   with the field's name;
//! - **answers**: s6-secrets-verify-pico on the fixture candidates reproduces
//!   `tests/parity/verify-expected.json` within 1e-5 (written by the native run with
//!   `SKY_VERIFY_UPDATE=1`), `judgeMany` answers as `judge`, and the stored thresholds are read;
//! - **the validation candidates** (native, only when `SKY_SECRETS2_DATA` names the data folder
//!   of s6-secrets v2): the state of every CredData validation candidate of the build's table,
//!   rebuilt from the generator's raw export, equals the table's, and with
//!   `SKY_SECRETS2_VAL_PROBS` (the model's validation dump, `val-creddata-probs.jsonl`) P(real)
//!   equals the dump's within 1e-5. Only counts are printed, never a value or a line.

use serde_json::{json, Value};
use sky_infer::secret_cand::WordSet;
use sky_infer::tokenizer::Tokenizer;
use sky_web::verify::{parse_candidate, state_of_json, Verifier};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

const FIXTURE: &str = include_str!("../../sky-infer/tests/fixtures/secret_cand.json");
const SMALL_TOKENIZER: &str =
    include_str!("../../sky-infer/tests/fixtures/small_tokenizer_s2.json");
const MODEL: &[u8] = include_bytes!("../../../site/public/models/s6-secrets-verify-pico.gguf");
const EXPECTED: &str = include_str!("parity/verify-expected.json");
const TOL: f64 = 1e-5;

fn fixture() -> Value {
    serde_json::from_str(FIXTURE).expect("secret_cand.json")
}

/// The fixture's word list, one entry per line.
fn fixture_words(f: &Value) -> String {
    f["dictionary"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| w.as_str().unwrap())
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn fixture_states_from_the_record() {
    let f = fixture();
    let tok = Tokenizer::from_tokenizer_json(SMALL_TOKENIZER).expect("tokenizer");
    let words = WordSet::from_text(&fixture_words(&f));
    let cands = f["candidates"].as_array().unwrap();
    assert!(cands.len() >= 10);
    for (i, c) in cands.iter().enumerate() {
        let want = c["state"].as_str().unwrap();
        // the record as the generator writes it, its file a real path
        let rec = c["record"].to_string();
        assert_eq!(state_of_json(&rec, &tok, &words).unwrap(), want, "{i}");
        // the same with an `ext` that disagrees: with `file` and no `scope`, both come from the file
        let mut r = c["record"].clone();
        r["ext"] = json!(".zz");
        assert_eq!(
            state_of_json(&r.to_string(), &tok, &words).unwrap(),
            want,
            "{i}"
        );
        // the scope and the extension given, no path
        let mut r = c["record"].clone();
        r.as_object_mut().unwrap().remove("file");
        r["scope"] = c["scope"].clone();
        r["ext"] = c["ext"].clone();
        assert_eq!(
            state_of_json(&r.to_string(), &tok, &words).unwrap(),
            want,
            "{i}"
        );
    }
}

#[test]
fn the_json_contract() {
    let f = fixture();
    let tok = Tokenizer::from_tokenizer_json(SMALL_TOKENIZER).expect("tokenizer");
    let words = WordSet::from_text(&fixture_words(&f));
    let state = |v: Value| state_of_json(&v.to_string(), &tok, &words);

    // the least a candidate can be: a rule and a value
    let s = state(json!({"rule": "r", "value": "abc"})).unwrap();
    let lines: Vec<&str> = s.lines().collect();
    assert_eq!(lines[0], "rule r");
    assert_eq!(lines[1], "path _");
    assert_eq!(lines[2], "key");
    assert_eq!(lines[3], "value abc");
    assert_eq!(&lines[5..], ["line <V>", "prev", "next"]);
    // null is absent; unknown fields are ignored
    let nulls = json!({"rule": "r", "value": "abc", "name": null, "file": null, "prevLine": null,
        "lineText": null, "filters": null, "entropy": null, "generator": "sg-cand-1", "cpStart": 4});
    assert_eq!(state(nulls).unwrap(), s);
    // a multi-line value without `endLine`: the line count comes from the value
    let multi =
        json!({"rule": "r", "value": "AAAA\n  BB", "lineText": "k: AAAA", "start": 3, "end": 4});
    let s = state(multi).unwrap();
    assert!(
        s.contains("\nline k: <V>\n"),
        "the value runs to the end of its first line"
    );
    let one = json!({"rule": "r", "value": "AAAA", "lineText": "k: AAAA;", "start": 3, "end": 7});
    assert!(state(one).unwrap().contains("\nline k: <V>;\n"));
    // filters
    let flt =
        json!({"rule": "r", "value": "abc", "filters": {"placeholder": true, "pragma": false}});
    assert!(state(flt)
        .unwrap()
        .contains(" gate 0 ph 1 stop 0 allow 0\n"));

    for (v, err) in [
        (json!([]), "a candidate is a JSON object"),
        (json!({"value": "abc"}), "`rule`: required"),
        (json!({"rule": "r"}), "`value`: required"),
        (
            json!({"rule": 3, "value": "abc"}),
            "`rule`: expected a string, got a number",
        ),
        (
            json!({"rule": "r", "value": "abc", "name": 1}),
            "`name`: expected a string",
        ),
        (
            json!({"rule": "r", "value": "abc", "lineText": "x abc", "start": 2}),
            "`start` and `end` are required",
        ),
        (
            json!({"rule": "r", "value": "abc", "lineText": "x abc", "start": 2, "end": 5.0}),
            "`end`: expected a non-negative integer, got another number",
        ),
        (
            json!({"rule": "r", "value": "abc", "lineText": "x abc", "start": -1, "end": 5}),
            "`start`: expected a non-negative integer",
        ),
        (
            json!({"rule": "r", "value": "abc", "lineFrom": "0"}),
            "`lineFrom`: expected a non-negative integer, got a string",
        ),
        (
            json!({"rule": "r", "value": "abc", "filters": {"entropy": 1}}),
            "`filters.entropy`: expected a boolean",
        ),
        (
            json!({"rule": "r", "value": "abc", "filters": {"context": "no"}}),
            "`filters.context`: expected a boolean",
        ),
        (
            json!({"rule": "r", "value": "abc", "filters": []}),
            "`filters`: expected an object",
        ),
        (
            json!({"rule": "r", "value": "abc", "entropy": "3.1"}),
            "`entropy`: expected a number",
        ),
        (
            json!({"rule": "r", "value": "abc", "scope": "test"}),
            "`scope`: expected an array",
        ),
        (
            json!({"rule": "r", "value": "abc", "scope": ["tests"]}),
            "`scope[0]`: not a word",
        ),
        (
            json!({"rule": "r", "value": "abc", "scope": ["test", 2]}),
            "`scope[1]`: expected a string",
        ),
        (
            json!({"rule": "r", "value": "abc", "masked": true}),
            "`masked`",
        ),
    ] {
        let e = state(v.clone()).unwrap_err();
        assert!(e.contains(err), "{v}: {e}");
    }
    assert!(state_of_json("{", &tok, &words).is_err());
    // scope words keep their slashes off, and the first four are written
    let c = parse_candidate(&json!({"rule": "r", "value": "v", "scope": ["test", "src", "conf", "docs", "lib"], "ext": ".py"})).unwrap();
    assert_eq!(c.scope, ["test", "src", "conf", "docs", "lib"]);
    let s = state(json!({"rule": "r", "value": "v", "scope": ["test", "src", "conf", "docs", "lib"], "ext": ".py"})).unwrap();
    assert_eq!(s.lines().nth(1), Some("path test src conf docs .py"));
}

/// The fixture's records and the verifier on the published model, with the fixture's words.
fn verifier() -> (Verifier, Vec<Value>) {
    let f = fixture();
    let v = Verifier::load(MODEL, fixture_words(&f).as_bytes()).expect("the verifier loads");
    let recs = f["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["record"].clone())
        .collect();
    (v, recs)
}

#[test]
fn answers_on_the_published_model() {
    let (mut v, recs) = verifier();
    let info = v.info();
    assert_eq!(info["window"], 320);
    assert_eq!(info["words"], 10);
    let t = v.thresholds().clone();
    for (k, want) in [
        ("tau_f1", 0.2598),
        ("tau_f05", 0.468437),
        ("tau_f2", 0.0780795),
    ] {
        let got = t[k].as_f64().unwrap_or_else(|| panic!("{k}"));
        // the card's values, rounded
        assert!((got - want).abs() < 1e-4 * want, "{k}: {got}");
    }
    let one: Vec<Value> = recs
        .iter()
        .map(|r| v.judge(&r.to_string()).expect("judge"))
        .collect();
    let many = v
        .judge_many(&Value::Array(recs.clone()).to_string())
        .expect("judgeMany");
    let many = many.as_array().unwrap();
    assert_eq!(many.len(), one.len());
    for (i, (a, b)) in one.iter().zip(many).enumerate() {
        assert_eq!(a["p_real"], b["p_real"], "{i}");
        assert_eq!(a["state_tokens"], b["state_tokens"], "{i}");
        let p = a["p_real"].as_f64().unwrap();
        assert!((0.0..=1.0).contains(&p), "{i}: {p}");
        let n = a["state_tokens"].as_u64().unwrap();
        assert_eq!(a["truncated"], json!(n > 320), "{i}");
        assert!(a["micros"].as_f64().unwrap() >= 0.0);
    }
    // a malformed candidate fails the batch with its index
    let mut bad = recs.clone();
    bad[2]["start"] = json!("3");
    let e = v.judge_many(&Value::Array(bad).to_string()).unwrap_err();
    assert!(e.starts_with("candidate 2: `start`"), "{e}");
    assert!(v.judge_many("{}").is_err());
    assert_eq!(v.judge_many("[]").unwrap(), json!([]));

    let got: Vec<Value> = one
        .iter()
        .map(|a| json!({"p_real": a["p_real"], "state_tokens": a["state_tokens"]}))
        .collect();
    #[cfg(not(target_arch = "wasm32"))]
    if std::env::var("SKY_VERIFY_UPDATE").is_ok_and(|v| v == "1") {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/parity/verify-expected.json"
        );
        std::fs::write(path, serde_json::to_string_pretty(&got).unwrap() + "\n").unwrap();
        return;
    }
    let want: Vec<Value> = serde_json::from_str(EXPECTED).expect("verify-expected.json");
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g["state_tokens"], w["state_tokens"], "{i}");
        let (a, b) = (g["p_real"].as_f64().unwrap(), w["p_real"].as_f64().unwrap());
        assert!((a - b).abs() <= TOL, "{i}: {a} vs expected {b}");
    }
}

#[test]
fn a_model_that_is_not_a_text_s2_model_is_refused() {
    let s1 = include_bytes!("../../../site/public/models/s1-pico.gguf");
    assert!(Verifier::load(s1, b"house\ntree\n").is_err());
    assert!(Verifier::load(MODEL, b"no\nab\n").is_err());
    assert!(Verifier::load(MODEL, &[0xff, 0xfe]).is_err());
}

/// The scope words of a CredData path (`data/<repo>/<scope…>/<file id><ext>`): its directories,
/// none for `_`, as the build reads them (`sky_data::secrets2::stored_scope`).
#[cfg(not(target_arch = "wasm32"))]
fn stored_scope(file: &str) -> Option<Vec<String>> {
    let parts: Vec<&str> = file.split('/').collect();
    if parts.len() < 4 || parts[0] != "data" {
        return None;
    }
    let dirs = &parts[2..parts.len() - 1];
    Some(if dirs == ["_"] {
        Vec::new()
    } else {
        dirs.iter().map(|d| d.to_string()).collect()
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn read_jsonl(path: &std::path::Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("a JSON line"))
        .collect()
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn validation_candidates_of_the_build() {
    use std::path::PathBuf;
    let Some(data) = std::env::var_os("SKY_SECRETS2_DATA").map(PathBuf::from) else {
        eprintln!("skipped: SKY_SECRETS2_DATA does not name the data folder of s6-secrets v2");
        return;
    };
    let words = std::fs::read(data.join("build/dictionary.txt")).expect("dictionary.txt");
    let mut v = Verifier::load(MODEL, &words).expect("the verifier loads");
    assert_eq!(v.info()["words"], 174_337);
    let raw = read_jsonl(&data.join("gen/candidates-validation.raw.jsonl"));
    let table = read_jsonl(&data.join("build/tables/creddata-val.jsonl"));
    let probs = std::env::var_os("SKY_SECRETS2_VAL_PROBS").map(|p| {
        read_jsonl(std::path::Path::new(&p))
            .into_iter()
            .filter(|d| d["src"] == "creddata:real")
            .map(|d| d["probs"][1].as_f64().expect("P(true)"))
            .collect::<Vec<f64>>()
    });
    if let Some(p) = &probs {
        assert_eq!(
            p.len(),
            table.len(),
            "one real question per row of the table"
        );
    }
    let (mut reachable, mut same, mut max_diff, mut truncated) = (0usize, 0usize, 0f64, 0usize);
    for (k, row) in table.iter().enumerate() {
        let id = row["id"].as_str().unwrap();
        let Some(rec) = id
            .strip_prefix("cred:val:")
            .and_then(|i| i.parse::<usize>().ok())
            .and_then(|i| raw.get(i))
        else {
            continue;
        };
        reachable += 1;
        let mut cand = rec.clone();
        let scope = stored_scope(rec["file"].as_str().unwrap()).expect("a CredData path");
        cand["scope"] = json!(scope);
        let cand = cand.to_string();
        let state = v.state(&cand).expect("state");
        if state == row["state"].as_str().unwrap() {
            same += 1;
        } else {
            // the label of the first part that differs, never its text
            let part = state
                .lines()
                .zip(row["state"].as_str().unwrap().lines())
                .find(|(a, b)| a != b)
                .and_then(|(a, _)| a.split(' ').next())
                .unwrap_or("?")
                .to_string();
            eprintln!("{id}: the state differs from part {part:?} on");
        }
        if let Some(p) = &probs {
            let a = v.judge(&cand).expect("judge");
            truncated += (a["truncated"] == json!(true)) as usize;
            max_diff = max_diff.max((a["p_real"].as_f64().unwrap() - p[k]).abs());
        }
    }
    eprintln!(
        "validation: {} rows, {reachable} reachable from the raw export, {same} states equal; \
         P(real) {}",
        table.len(),
        match &probs {
            Some(_) => format!("max |diff| {max_diff:.3e} on {reachable}, {truncated} truncated"),
            None => "not checked (SKY_SECRETS2_VAL_PROBS unset)".into(),
        }
    );
    assert!(reachable > 0);
    assert_eq!(same, reachable);
    assert!(max_diff <= TOL, "max |diff| {max_diff}");
}
