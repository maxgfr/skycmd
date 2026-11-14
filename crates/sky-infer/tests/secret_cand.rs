//! The state of a rule candidate (`sky_infer::secret_cand`, `docs/s6-secrets2-prereg.md`
//! section 4) against the shared vectors of `tests/fixtures/secret_cand.json`: path scopes,
//! hand features and full states that every implementation (the trainer's data tool, the engine,
//! the TypeScript host) must reproduce, as `num_extract.json` does for numbers.
//!
//! The file's layout:
//!   tokenizer    the tokenizer of the `tokens` and `charsPerToken` features, a file of this
//!                folder (a deployed model uses its own)
//!   dictionary   the word list of the `wordRun` feature, one entry per element
//!   wordInPath   CredData's `word_in_path.json`
//!   scopes       [{path, scope, words, ext}]: `scope` is CredData's `get_file_scope()` of the
//!                path without its extension, `words` its directories (none for `/_/`)
//!   candidates   [{record, scope, ext, features, state}]: `record` has the fields of an
//!                `sg-cand-1` candidate record with `file` a real path; `scope` holds every
//!                scope word (a state writes the first four)
//!
//! The inputs are assembled here, in code: every value is invented and none follows a provider's
//! format. `SKY_WRITE_FIXTURES=1 cargo test -p sky-infer --test secret_cand` writes the file
//! again from this implementation; without it the test checks that the file holds these inputs
//! and that the outputs still match.

use std::path::PathBuf;

use serde_json::{json, Value};
use sky_infer::secret_cand::{
    file_scope, path_scope, scope_words, state_with, value_columns, Candidate, Features, Filters,
    WordSet, ELISION_MARK, MAX_SCOPE_WORDS, VALUE_MARK, WORD_IN_PATH,
};
use sky_infer::tokenizer::Tokenizer;

const TOKENIZER: &str = "small_tokenizer_s2.json";

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// A small linear congruential generator: the invented values are the same on every run.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    fn string(&mut self, set: &str, n: usize) -> String {
        let set: Vec<char> = set.chars().collect();
        (0..n)
            .map(|_| set[self.next() as usize % set.len()])
            .collect()
    }
}

const HEX: &str = "0123456789abcdef";
const ALNUM: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
const B64: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

const DICTIONARY: [&str; 12] = [
    "battery",
    "correct",
    "horse",
    "house",
    "password",
    "secret",
    "staple",
    "sunflower",
    "tree",
    "winter",
    "Paris",
    "cat",
];

const PATHS: [&str; 22] = [
    "/usr/local/lib/python",
    "config/config.yml",
    "pet/modules/stdlib/spec/functions/t.rb",
    "src/test/resources/dockerConfig/myDockerCfg",
    "testapi/src/java/TestApiExample.java",
    "packages/src/lib/__tests__/fixtures/request.json",
    "X3",
    "",
    "README.md",
    ".env",
    "deploy/.env.production",
    "docs/guide/setup.MD",
    "doc/api/index.rst",
    "documentation/intro.txt",
    "app/models/user.rb",
    "vendor/github.com/acme/client/client.go",
    "SRC/Main/Resources/Application.PROPERTIES",
    "other/public/assets/site.js",
    "k8s/kube-system/cluster/templates/secret.yaml",
    "build/scripts/init.sh",
    "tools/fuzz/corpus/seed-0001",
    "lib/python3.12/site-packages/pkg/settings.py",
];

/// One candidate record: `file`, `rule`, `name`, the lines around the value, the value, and
/// where it is written in its line (`before` it and `after` it on the line).
#[allow(clippy::too_many_arguments)]
fn record(
    file: &str,
    rule: &str,
    name: Option<&str>,
    prev: Option<&str>,
    before: &str,
    value: &str,
    after: &str,
    next: Option<&str>,
    filters: [bool; 4],
) -> Value {
    let units = |s: &str| s.encode_utf16().count();
    let first = value.split('\n').next().unwrap_or("");
    let lines = value.split('\n').count() as u64;
    let last = value.rsplit('\n').next().unwrap_or("");
    let line_text = if lines > 1 {
        format!("{before}{first}")
    } else {
        format!("{before}{value}{after}")
    };
    let end = if lines > 1 {
        units(last)
    } else {
        units(before) + units(value)
    };
    json!({
        "generator": "sg-cand-1",
        "revision": 1,
        "file": file,
        "rule": rule,
        "line": 7,
        "start": units(before),
        "end": end,
        "endLine": 7 + lines - 1,
        "length": units(value),
        "value": value,
        "name": name,
        "lineText": line_text,
        "lineFrom": 0,
        "prevLine": prev,
        "nextLine": next,
        "filters": {
            "entropy": filters[0],
            "placeholder": filters[1],
            "stopword": filters[2],
            "allowlist": filters[3],
            "context": false,
            "pragma": false,
        },
    })
}

/// The same record as the generator writes it for a long line: a window of the line that
/// starts at column `from`.
fn windowed(mut rec: Value, from: usize, len: usize) -> Value {
    let line: Vec<u16> = rec["lineText"].as_str().unwrap().encode_utf16().collect();
    let to = (from + len).min(line.len());
    rec["lineText"] = json!(String::from_utf16(&line[from..to]).unwrap());
    rec["lineFrom"] = json!(from);
    rec
}

fn records() -> Vec<Value> {
    let mut g = Lcg(20_261_114);
    let none = [false; 4];
    let hex32 = g.string(HEX, 32);
    let hex40 = g.string(HEX, 40);
    let alnum24 = g.string(ALNUM, 24);
    let long = g.string(B64, 200);
    let exactly64 = g.string(ALNUM, 64);
    let just65 = g.string(ALNUM, 65);
    let uuid = format!(
        "{}-{}-4{}-a{}-{}",
        g.string(HEX, 8),
        g.string(HEX, 4),
        g.string(HEX, 3),
        g.string(HEX, 3),
        g.string(HEX, 12)
    );
    let wide_left = format!("{}  token = \"", "x = 1; ".repeat(40));
    let wide_right = format!("\"; {}", "y += 2; ".repeat(40));
    let block = format!(
        "{}\n{}\n{}",
        g.string(B64, 48),
        g.string(B64, 48),
        g.string(B64, 20)
    );
    let long_name = format!("service.{}.client_secret_value", "nested.".repeat(8));
    vec![
        record(
            "src/app/settings.py",
            "generic-keyword",
            Some("db_password"),
            Some("import os"),
            "db_password = \"",
            "Sunflower42!",
            "\"",
            Some("connect(db_password)"),
            none,
        ),
        record(
            "tests/fixtures/config/app.yml",
            "generic-api-key",
            Some("api_key"),
            Some("service:"),
            "  api_key: ",
            &hex32,
            "",
            Some("  region: eu-west-1"),
            [false, false, false, true],
        ),
        record(
            "config/deploy.JSON",
            "generic-keyword",
            Some(&long_name),
            Some("{"),
            &format!("  \"{long_name}\": \""),
            &long,
            "\",",
            Some("  \"retries\": 3"),
            none,
        ),
        record(
            "pkg/server/handler.go",
            "generic-keyword",
            Some("token"),
            None,
            &wide_left,
            &alnum24,
            &wide_right,
            None,
            [true, false, true, false],
        ),
        windowed(
            record(
                "pkg/server/handler.go",
                "generic-keyword",
                Some("token"),
                Some("// generated"),
                &wide_left,
                &alnum24,
                &wide_right,
                Some("}"),
                none,
            ),
            211,
            200,
        ),
        record(
            "deploy/keys/service.pem",
            "private-key",
            None,
            Some("# rotated yearly"),
            "key: |",
            &block,
            "",
            Some("owner: ops"),
            none,
        ),
        record(
            "Makefile",
            "generic-keyword",
            None,
            None,
            "",
            "correct-horse-battery-staple",
            "",
            None,
            none,
        ),
        record(
            "src/i18n/messages.properties",
            "generic-keyword",
            Some("mot_de_passe"),
            Some("# accès"),
            "\u{1F511} mot_de_passe = ",
            "pässwörd-Été2024\u{1F512}",
            " # à changer",
            Some("utilisateur = rené"),
            none,
        ),
        record(
            "docs/README.md",
            "generic-keyword",
            Some("API_TOKEN"),
            Some("Set your token:"),
            "export API_TOKEN=",
            "<your-api-token>",
            "",
            Some("Then run the tool."),
            [false, true, false, false],
        ),
        record(
            ".env.example",
            "generic-api-key",
            Some("SECRET_KEY"),
            Some("DEBUG=false"),
            "SECRET_KEY=",
            "xxxxxxxxxxxxxxxxxxxxxxxx",
            "",
            Some("PORT=8080"),
            [true, true, true, true],
        ),
        record(
            "lib/checks.rb",
            "generic-keyword",
            Some("cache_key"),
            Some("def digest"),
            "  cache_key = '",
            "aaaaaaaabbbbccde",
            "'",
            Some("end"),
            [true, false, false, false],
        ),
        record(
            "lib/checks.rb",
            "generic-keyword",
            Some("request_token"),
            Some("def trace"),
            "  request_token = '",
            &uuid,
            "'",
            Some("end"),
            none,
        ),
        record(
            "src/test/resources/dockerConfig/secure/util/myDockerCfg.json",
            "generic-keyword",
            Some("auth"),
            Some("\"registry.example.com\": {"),
            "    \"auth\": \"",
            &hex40,
            "\"",
            Some("}"),
            none,
        ),
        record(
            "app/models/account.rb",
            "generic-keyword",
            Some("pin_secret"),
            Some("  # 4 digits, then a version"),
            "  pin_secret = ",
            "2024.5-17",
            "",
            Some("  retries = -3"),
            [true, false, false, false],
        ),
        record(
            "notes.txt",
            "generic-keyword",
            Some("pass"),
            Some(""),
            "pass: ",
            "",
            "",
            Some(""),
            none,
        ),
        record(
            "internal/tool/gen.py",
            "generic-keyword",
            Some("SALT"),
            Some("SALT_ROUNDS = 12"),
            "SALT = b\"",
            &exactly64,
            "\"",
            Some("def derive(password):"),
            none,
        ),
        record(
            "internal/tool/gen.py",
            "generic-keyword",
            Some("NONCE"),
            Some("import secrets"),
            "NONCE = \"",
            &just65,
            "\"",
            Some("def sign(body):"),
            none,
        ),
        record(
            "server/routes/login.ts",
            "auth-scheme",
            None,
            Some("  headers: {"),
            "    Authorization: `Bearer ",
            "Winter-house_7.tree",
            "`,",
            Some("  },"),
            [false, false, false, true],
        ),
        record(
            "",
            "issue-regex",
            None,
            Some("I ran the command below and it failed."),
            "curl -u admin:",
            "Password123",
            " https://example.invalid/api",
            Some("It returns 401."),
            none,
        ),
    ]
}

fn string_or_none(v: &Value) -> Option<&str> {
    v.as_str()
}

/// What this implementation gives for one record: `(scope words, ext, features, state)`.
type Outputs = (Vec<&'static str>, String, Features, String);

fn outputs(rec: &Value, tok: &Tokenizer, words: &WordSet) -> Outputs {
    let (scope, ext) = path_scope(rec["file"].as_str().unwrap());
    let flag = |k: &str| rec["filters"][k].as_bool().unwrap();
    let filters = Filters {
        entropy: flag("entropy"),
        placeholder: flag("placeholder"),
        stopword: flag("stopword"),
        allowlist: flag("allowlist"),
    };
    let n = |k: &str| rec[k].as_u64().unwrap();
    let (value_start, value_end) = value_columns(
        n("line"),
        n("endLine"),
        n("start") as usize,
        n("end") as usize,
        n("lineFrom") as usize,
    );
    let value = rec["value"].as_str().unwrap();
    let c = Candidate {
        rule: rec["rule"].as_str().unwrap(),
        scope: &scope,
        ext: &ext,
        name: string_or_none(&rec["name"]),
        value,
        line_text: rec["lineText"].as_str().unwrap(),
        value_start,
        value_end,
        prev_line: string_or_none(&rec["prevLine"]),
        next_line: string_or_none(&rec["nextLine"]),
        filters,
    };
    let f = Features::of(value, tok, words, filters);
    let state = state_with(&c, &f);
    (scope.clone(), ext.clone(), f, state)
}

fn features_json(f: &Features) -> Value {
    json!({
        "len": f.len,
        "entropy": f.entropy,
        "tokens": f.tokens,
        "charsPerToken": f.chars_per_token,
        "digits": f.digits,
        "upper": f.upper,
        "lower": f.lower,
        "other": f.other,
        "wordRun": f.word_run,
        "gate": f.filters.entropy,
        "placeholder": f.filters.placeholder,
        "stopword": f.filters.stopword,
        "allowlist": f.filters.allowlist,
    })
}

fn build(tok: &Tokenizer, words: &WordSet) -> Value {
    let scopes: Vec<Value> = PATHS
        .iter()
        .map(|p| {
            let (stem, ext) = sky_infer::secret_cand::split_extension(p);
            json!({"path": p, "scope": file_scope(stem), "words": scope_words(stem), "ext": ext})
        })
        .collect();
    let candidates: Vec<Value> = records()
        .into_iter()
        .map(|rec| {
            let (scope, ext, f, state) = outputs(&rec, tok, words);
            json!({"record": rec, "scope": scope, "ext": ext, "features": features_json(&f), "state": state})
        })
        .collect();
    json!({
        "tokenizer": TOKENIZER,
        "dictionary": DICTIONARY,
        "wordInPath": WORD_IN_PATH.as_slice(),
        "scopes": scopes,
        "candidates": candidates,
    })
}

fn setup() -> (Tokenizer, WordSet) {
    let json = std::fs::read_to_string(fixtures().join(TOKENIZER)).unwrap();
    let tok = Tokenizer::from_tokenizer_json(&json).unwrap();
    (tok, WordSet::from_text(&DICTIONARY.join("\n")))
}

#[test]
fn matches_the_shared_vectors() {
    let (tok, words) = setup();
    let path = fixtures().join("secret_cand.json");
    let built = build(&tok, &words);
    if std::env::var_os("SKY_WRITE_FIXTURES").is_some() {
        let text = serde_json::to_string_pretty(&built).unwrap() + "\n";
        std::fs::write(&path, text).unwrap();
    }
    let Ok(bytes) = std::fs::read(&path) else {
        panic!("{} not found", path.display());
    };
    let file: Value = serde_json::from_slice(&bytes).unwrap();
    // the file's own inputs, read back: what another implementation is given
    assert_eq!(file["tokenizer"], TOKENIZER);
    let dict: Vec<String> = serde_json::from_value(file["dictionary"].clone()).unwrap();
    let words = WordSet::from_text(&dict.join("\n"));
    let list: Vec<String> = serde_json::from_value(file["wordInPath"].clone()).unwrap();
    assert_eq!(list, WORD_IN_PATH);
    let scopes = file["scopes"].as_array().unwrap();
    assert!(scopes.len() >= 20);
    for s in scopes {
        let p = s["path"].as_str().unwrap();
        let (stem, ext) = sky_infer::secret_cand::split_extension(p);
        assert_eq!(file_scope(stem), s["scope"].as_str().unwrap(), "{p}");
        assert_eq!(json!(scope_words(stem)), s["words"], "{p}");
        assert_eq!(ext, s["ext"].as_str().unwrap(), "{p}");
    }
    let cands = file["candidates"].as_array().unwrap();
    assert!(cands.len() >= 18);
    for (i, c) in cands.iter().enumerate() {
        let (scope, ext, f, state) = outputs(&c["record"], &tok, &words);
        assert_eq!(json!(scope), c["scope"], "candidate {i}");
        assert_eq!(ext, c["ext"].as_str().unwrap(), "candidate {i}");
        assert_eq!(features_json(&f), c["features"], "candidate {i}");
        assert_eq!(state, c["state"].as_str().unwrap(), "candidate {i}");
    }
    // the file was written from the inputs assembled above
    assert_eq!(file, built, "tests/fixtures/secret_cand.json is stale");
}

#[test]
fn the_vectors_cover_the_cases_that_matter() {
    let (tok, words) = setup();
    let all: Vec<_> = records()
        .iter()
        .map(|r| (r.clone(), outputs(r, &tok, &words)))
        .collect();
    let states: Vec<&str> = all.iter().map(|(_, o)| o.3.as_str()).collect();
    // a state has its eight parts in the order of section 4.1, the features as one line
    for s in &states {
        let labels: Vec<&str> = [
            "rule ", "\npath ", "\nkey", "\nvalue", "\nlen ", "\nline ", "\nprev", "\nnext",
        ]
        .to_vec();
        let mut at = 0;
        for l in labels {
            let i = s[at..].find(l).unwrap_or_else(|| panic!("{l:?} in {s:?}"));
            at += i + l.len();
        }
    }
    let has = |p: &dyn Fn(&(Value, Outputs)) -> bool| all.iter().any(p);
    // a long value is elided, a 64-character one is not
    assert!(has(&|(r, o)| r["length"] == 200
        && o.3.contains(ELISION_MARK)
        && o.2.len == 200));
    assert!(has(
        &|(r, o)| r["length"] == 64 && !o.3.contains(ELISION_MARK)
    ));
    assert!(has(
        &|(r, o)| r["length"] == 65 && o.3.contains(ELISION_MARK)
    ));
    // more than four scope words: four are written
    assert!(has(&|(_, o)| o.0.len() > MAX_SCOPE_WORDS
        && o.3.contains(&format!("\npath {} ", o.0[..4].join(" ")))));
    // no scope, no extension, no name, no neighbours
    assert!(has(
        &|(_, o)| o.3.contains("\npath _\nkey\nvalue ") && o.3.ends_with("\nprev\nnext")
    ));
    // a window of a long line gives the state of the whole line
    let wide: Vec<&str> = all
        .iter()
        .filter(|(r, _)| r["file"] == "pkg/server/handler.go")
        .map(|(_, o)| {
            o.3.split("\nline ")
                .nth(1)
                .unwrap()
                .split('\n')
                .next()
                .unwrap()
        })
        .collect();
    assert_eq!(wide.len(), 2);
    assert_eq!(wide[0], wide[1]);
    assert_eq!(wide[0].chars().count(), 120);
    // a value over several lines: its first line ends at the mark
    assert!(has(&|(r, o)| r["endLine"] != r["line"]
        && o.3.contains(&format!("\nline key: |{VALUE_MARK}\nprev "))));
    // every flag is set somewhere, and the rounding tie is in
    assert!(has(&|(_, o)| o.3.contains(" gate 1 ph 1 stop 1 allow 1")));
    assert!(has(&|(_, o)| o.2.entropy == 188));
    assert!(has(&|(_, o)| o.2.word_run == 9) && has(&|(_, o)| o.2.word_run == 7));
    assert!(has(&|(_, o)| o.2.len == 0 && o.2.chars_per_token == 0));
    assert!(has(&|(_, o)| o.2.other >= 4 && o.2.tokens > o.2.len));
}
