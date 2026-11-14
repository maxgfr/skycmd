//! Synthetic candidates (`docs/s6-secrets2-prereg.md`, section 3.3): `sky-data secret-cand`.
//!
//! The mode writes candidate records in the format of the generator `sg-cand-1`, so that the one
//! state builder serves them, but they do not come from it: nothing is scanned. Each record is a
//! value placed in a few invented lines of a file with an invented path, with the rule id the
//! engine would most plausibly name it by ([`rule_of`]) and the two filter flags that can be
//! computed without the engine ([`flags_of`]: its entropy thresholds and an approximation of
//! its placeholder test; the stopword and allowlist flags are never set). Every key follows a
//! public provider format with a random body; nothing is real.
//!
//! **Positives** (`T`): provider-format keys under their usual names, and word-based passwords
//! (one or two dictionary words with digits and a symbol) under password-like names, which
//! CredData's obfuscation cannot provide.
//!
//! **Hard negatives**, each a class of [`CLASSES`]: test keys and publishable keys;
//! placeholders, masked values, environment and template references; public keys and
//! certificates; checksums and integrity hashes; UUIDs; git commit ids; base64 image or font
//! data; documentation lines; test fixtures under a `test` scope; and near-miss pairs, the same
//! lines once with a random secret (`T`) and once with a placeholder or a reference (`X`).
//!
//! The synthetic set is used for training and for a validation sample only; it enters no test.

use std::collections::HashSet;

use anyhow::Result;
use sky_infer::secret_cand::{entropy, WordSet};

use super::{record_at, Item, Label, RecFilters, SYNTH};
use crate::rng::{hash64, Rng};
use crate::secrets::{
    camel, live, lower, pick, rand_str, rand_var, stripe_body, Kind, Template, Value, ALNUM, B64,
    BENCH_TEMPLATES, DIGITS, HEX, SERVICES, TRAIN_TEMPLATES,
};

/// The `generator` field of a synthetic record.
pub const GENERATOR: &str = "sky-secret-cand";

/// Synthetic training candidates asked for when the cap allows (section 3.4: "about 20,000").
pub const TRAIN_TARGET: usize = 20_000;
/// Synthetic validation candidates (section 3.4).
pub const VAL_CANDIDATES: usize = 2_000;

/// The classes, their weight in a hundred draws and what a draw gives. A near-miss draw gives
/// two candidates.
pub const CLASSES: [(&str, usize); 13] = [
    ("provider-key", 22),
    ("word-password", 14),
    ("near-miss", 8),
    ("test-key", 4),
    ("publishable-key", 3),
    ("placeholder", 12),
    ("public-key", 5),
    ("checksum", 7),
    ("uuid", 5),
    ("commit-id", 4),
    ("base64-data", 4),
    ("documentation", 6),
    ("test-fixture", 6),
];

const BASE32: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
const SYMBOLS: &[u8] = b"!@#$%&*?-_+.";

/// An AWS access key id as AWS issues them: `AKIA` and 16 base32 characters (`A-Z`, `2-7`).
/// The generator of s6-secrets-pico (`crate::secrets::live`) draws the body over `A-Z0-9`; it is
/// left as it is so that the published bench stays reproducible, and this mode does not use it.
pub fn aws_access_key_id(rng: &mut Rng) -> String {
    format!("AKIA{}", rand_str(rng, BASE32, 16))
}

/// A live provider secret and its usual variable name: `crate::secrets::live`, with the AWS
/// access key id drawn again in its real alphabet.
fn provider_secret(rng: &mut Rng) -> (String, String) {
    let svc = SERVICES[rng.below(SERVICES.len() - 1)].0;
    let (text, var) = live(rng, svc);
    if svc == "aws" && text.starts_with("AKIA") {
        return (aws_access_key_id(rng), var.to_string());
    }
    (text, var.to_string())
}

/// The rule id the engine would most plausibly give a value under a name. An approximation by
/// prefix and form: the synthetic records are not the generator's output.
pub fn rule_of(value: &str, name: Option<&str>, quoted: bool) -> &'static str {
    const BY_PREFIX: [(&str, &str); 22] = [
        ("AKIA", "aws-access-token"),
        ("github_pat_", "github-fine-grained-pat"),
        ("ghp_", "github-pat"),
        ("gho_", "github-oauth"),
        ("ghu_", "github-app-token"),
        ("ghs_", "github-app-token"),
        ("glpat-", "gitlab-pat"),
        ("sk_live_", "stripe-access-token"),
        ("rk_live_", "stripe-access-token"),
        ("sk_test_", "stripe-access-token"),
        ("rk_test_", "stripe-access-token"),
        ("xoxb-", "slack-bot-token"),
        ("xoxp-", "slack-user-token"),
        ("https://hooks.slack.com/", "slack-webhook-url"),
        ("sk-ant-api03-", "anthropic-api-key"),
        ("AIza", "gcp-api-key"),
        ("hf_", "huggingface-access-token"),
        ("SG.", "sendgrid-api-token"),
        ("npm_", "npm-access-token"),
        ("-----BEGIN ", "private-key"),
        ("eyJ", "jwt"),
        ("sk-proj-", "generic-api-key"),
    ];
    if let Some((_, rule)) = BY_PREFIX.iter().find(|(p, _)| value.starts_with(p)) {
        if *rule != "private-key" || value.contains("PRIVATE KEY") {
            return rule;
        }
    }
    let name = name.unwrap_or("").to_ascii_lowercase();
    let has = |words: &[&str]| words.iter().any(|w| name.contains(w));
    let plain = value
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'=' | b'-'));
    if plain
        && (10..=150).contains(&value.len())
        && has(&[
            "access",
            "api",
            "auth",
            "key",
            "credential",
            "creds",
            "passwd",
            "password",
            "secret",
            "token",
        ])
    {
        return "generic-api-key";
    }
    if quoted
        && (6..=200).contains(&value.len())
        && !value.contains(char::is_whitespace)
        && has(&[
            "password",
            "passwd",
            "pwd",
            "secret",
            "access_key",
            "accesskey",
            "api_key",
            "apikey",
            "auth_token",
        ])
    {
        return "password-assignment";
    }
    "generic-keyword"
}

/// The entropy threshold of a rule in the pinned rules (gitleaks `09242ce`, secretgate's
/// built-in rules), for the rules [`rule_of`] names.
fn entropy_threshold(rule: &str) -> Option<f64> {
    Some(match rule {
        "generic-api-key" => 3.5,
        "gcp-api-key" => 4.0,
        "password-assignment"
        | "aws-access-token"
        | "github-pat"
        | "github-oauth"
        | "github-app-token"
        | "github-fine-grained-pat"
        | "gitlab-pat"
        | "slack-bot-token"
        | "jwt" => 3.0,
        "stripe-access-token"
        | "slack-user-token"
        | "huggingface-access-token"
        | "sendgrid-api-token"
        | "npm-access-token" => 2.0,
        _ => return None,
    })
}

/// A value the default engine's placeholder test would most plausibly drop: masked, an
/// interpolation, an angle-bracket slot, the documentation ending `EXAMPLE`, a few repeated
/// letters, or a word that only describes what belongs there.
pub fn looks_like_placeholder(v: &str) -> bool {
    const WORDS: [&str; 16] = [
        "password",
        "passwd",
        "pass",
        "secret",
        "changeme",
        "change-me",
        "example",
        "test",
        "token",
        "your_password",
        "dummy",
        "todo",
        "replace-me",
        "placeholder",
        "your-api-key-here",
        "your_api_key",
    ];
    let low = v.to_ascii_lowercase();
    let all = |f: &dyn Fn(char) -> bool| !v.is_empty() && v.chars().all(f);
    let wrapped =
        |a: &str, b: &str| v.len() > a.len() + b.len() && v.starts_with(a) && v.ends_with(b);
    WORDS.contains(&low.as_str())
        || all(&|c| matches!(c, '*' | 'x' | '.' | '-' | '_'))
        || v.contains("****")
        || (v.ends_with("EXAMPLE")
            && v.len() > 7
            && v.as_bytes()[v.len() - 8].is_ascii_alphanumeric()
            && !v.as_bytes()[v.len() - 8].is_ascii_lowercase())
        || wrapped("${", "}")
        || wrapped("{{", "}}")
        || wrapped("<", ">")
        || wrapped("%", "%")
        || (v.starts_with('$')
            && v.len() > 1
            && v[1..]
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_'))
        || (all(&|c| c.is_ascii_lowercase()) && v.chars().collect::<HashSet<_>>().len() <= 3)
}

/// The filter flags a synthetic record carries.
pub fn flags_of(rule: &str, value: &str) -> RecFilters {
    RecFilters {
        entropy: entropy_threshold(rule).is_some_and(|t| entropy(value) <= t),
        placeholder: looks_like_placeholder(value),
        ..RecFilters::default()
    }
}

/// Stands for the value while a context is drawn, so that two values can be put in the same
/// lines.
const SLOT: &str = "\u{1}SLOT\u{1}";

/// A context: its lines with [`SLOT`] where the value goes, the line of the slot, and the file.
#[derive(Clone, Debug)]
struct Context {
    lines: Vec<String>,
    at: usize,
    file: String,
}

fn all_templates() -> Vec<(&'static str, Template)> {
    TRAIN_TEMPLATES
        .iter()
        .chain(&BENCH_TEMPLATES)
        .copied()
        .collect()
}

/// A file a context of `template` could be, outside any test or documentation directory.
fn source_file(rng: &mut Rng, template: &str) -> String {
    let files: &[&str] = match template {
        "env" => &[
            ".env",
            "config/.env.production",
            "deploy/prod.env",
            ".env.local",
            "docker/app.env",
        ],
        "export" => &[
            "scripts/deploy.sh",
            "bin/release.sh",
            "ci/setup.sh",
            ".envrc",
        ],
        "python" => &[
            "src/app/settings.py",
            "app/config.py",
            "scripts/sync.py",
            "lib/client.py",
        ],
        "js" => &[
            "src/config.js",
            "server/index.js",
            "lib/api/client.ts",
            "app/services/auth.ts",
        ],
        "yaml" => &[
            "config/app.yml",
            "deploy/values.yaml",
            "k8s/secret.yaml",
            "ansible/group_vars/all.yml",
        ],
        "json" => &[
            "config/default.json",
            "appsettings.json",
            "conf/service.json",
            "deploy/params.json",
        ],
        "curl" => &["scripts/smoke.sh", "bin/check-api.sh", "ops/runbook.sh"],
        "log" => &["logs/app.log", "var/log/service.log", "ci/output.log"],
        "diff" => &["patches/session.diff", "changes.patch"],
        "compose" => &["docker-compose.yml", "deploy/docker-compose.prod.yml"],
        "toml" => &["config.toml", "conf/settings.toml", "deploy/app.toml"],
        "go" => &[
            "cmd/server/main.go",
            "internal/client/client.go",
            "pkg/api/auth.go",
        ],
        _ => &["setup.cfg", "conf/mail.ini", "etc/app.conf"],
    };
    pick(rng, files).to_string()
}

/// The same file under a test scope.
fn test_file(rng: &mut Rng, file: &str) -> String {
    let name = file.rsplit('/').next().unwrap_or(file);
    let dir = pick(
        rng,
        &[
            "tests/fixtures",
            "test/data",
            "spec/support",
            "src/test/resources",
            "__tests__/mocks",
            "testdata",
            "tests/unit",
            "e2e/fixtures",
        ],
    );
    format!("{dir}/{name}")
}

/// The same file under a documentation scope.
fn docs_file(rng: &mut Rng) -> String {
    pick(
        rng,
        &[
            "README.md",
            "docs/setup.md",
            "docs/api/authentication.md",
            "doc/guide/quickstart.rst",
            "examples/README.md",
            "docs/configuration.md",
            "CONTRIBUTING.md",
        ],
    )
    .to_string()
}

/// A context from one of the generator's templates, for a value under `var`.
fn template_context(rng: &mut Rng, var: &str) -> Context {
    let templates = all_templates();
    let (name, t) = templates[rng.below(templates.len())];
    let v = Value {
        text: SLOT.to_string(),
        var: var.to_string(),
        kind: Kind::Secret,
        service: 0,
    };
    let (lines, at) = t(rng, &v);
    Context {
        lines,
        at,
        file: source_file(rng, name),
    }
}

/// A context of given lines.
fn lines_context(file: &str, before: &[&str], line: String, after: &[&str]) -> Context {
    let mut lines: Vec<String> = before.iter().map(|s| s.to_string()).collect();
    let at = lines.len();
    lines.push(line);
    lines.extend(after.iter().map(|s| s.to_string()));
    Context {
        lines,
        at,
        file: file.to_string(),
    }
}

/// The name a value is assigned to on its line: the last run of name characters before the
/// value, past the quotes, the separator and an authentication scheme word.
fn name_before(before: &str) -> Option<String> {
    let mut s = before.trim_end_matches(['"', '\'', '`', ' ', '\t', '[']);
    for scheme in ["Bearer", "Basic", "bearer", "basic", "token", "Token"] {
        if let Some(rest) = s.strip_suffix(scheme) {
            if rest.ends_with([' ', ':', '=', '"', '\'']) {
                s = rest.trim_end_matches(['"', '\'', '`', ' ', '\t']);
            }
        }
    }
    let s = s.trim_end_matches(['=', ':', '>', '!']);
    let s = s.trim_end_matches(['"', '\'', '`', ' ', '\t', ']', ')']);
    let n = s
        .bytes()
        .rev()
        .take_while(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'$' | b'-'))
        .count();
    (n > 0).then(|| s[s.len() - n..].to_string())
}

/// One synthetic candidate: `value` in the slot of `cx`.
fn candidate(cx: &Context, value: &str, label: Label, class: &str) -> Option<Item> {
    let col = cx.lines[cx.at].find(SLOT)?;
    let before = &cx.lines[cx.at][..col];
    let name = name_before(before);
    let quoted = before.ends_with(['"', '\'', '`']);
    let text = cx.lines.join("\n").replace(SLOT, value);
    let start: usize = cx.lines[..cx.at].iter().map(|l| l.len() + 1).sum::<usize>() + col;
    // the value of a connection string is its password
    let password = value.find("://").and_then(|i| {
        let rest = &value[i + 3..];
        let (c, a) = (rest.find(':')?, rest.find('@')?);
        (c + 1 < a).then_some((i + 3 + c + 1, &rest[c + 1..a]))
    });
    let (start, value, rule) = match password {
        Some((off, pass)) => (start + off, pass, "url-credentials"),
        None => (start, value, rule_of(value, name.as_deref(), quoted)),
    };
    let rec = record_at(
        GENERATOR,
        Some(&cx.file),
        rule,
        name.as_deref(),
        &text,
        start,
        start + value.len(),
        flags_of(rule, value),
    );
    Some(Item {
        id: String::new(),
        source: SYNTH.to_string(),
        split: String::new(),
        label,
        weak: false,
        kind: true,
        class: Some(class.to_string()),
        via: None,
        rec,
    })
}

const PASSWORD_VARS: [&str; 16] = [
    "PASSWORD",
    "DB_PASSWORD",
    "DB_PASS",
    "ADMIN_PASSWORD",
    "MYSQL_ROOT_PASSWORD",
    "POSTGRES_PASSWORD",
    "REDIS_PASSWORD",
    "SMTP_PASSWORD",
    "PASSWD",
    "USER_PWD",
    "LOGIN_PASSWORD",
    "ROOT_PASS",
    "FTP_PASSWORD",
    "LDAP_BIND_PASSWORD",
    "KEYSTORE_PASSWORD",
    "SECRET",
];

const SECRET_VARS: [&str; 14] = [
    "API_KEY",
    "API_SECRET",
    "ACCESS_TOKEN",
    "AUTH_TOKEN",
    "CLIENT_SECRET",
    "SECRET_KEY",
    "SERVICE_TOKEN",
    "PRIVATE_TOKEN",
    "APP_SECRET",
    "TOKEN",
    "WEBHOOK_SECRET",
    "SIGNING_KEY",
    "ENCRYPTION_KEY",
    "SESSION_SECRET",
];

/// Names of values that are not credentials and still hold a keyword of the candidate rule.
const NEUTRAL_KEY_VARS: [&str; 12] = [
    "CACHE_KEY",
    "IDEMPOTENCY_KEY",
    "PARTITION_KEY",
    "SORT_KEY",
    "KEY_ID",
    "TOKEN_ID",
    "API_VERSION",
    "KEY_FINGERPRINT",
    "PRIMARY_KEY",
    "ETAG_KEY",
    "AUTHOR_ID",
    "PASSAGE_ID",
];

/// Words a word-based password is not built from: they are what a placeholder says.
const NOT_A_PASSWORD_WORD: [&str; 12] = [
    "password",
    "secret",
    "example",
    "test",
    "token",
    "sample",
    "dummy",
    "demo",
    "default",
    "changeme",
    "placeholder",
    "testing",
];

/// A word of 4 to 8 letters in one of three cases.
fn word(rng: &mut Rng, words: &WordSet) -> String {
    loop {
        let pool = words.of_len(4 + rng.below(5));
        if pool.is_empty() {
            continue;
        }
        let w = &pool[rng.below(pool.len())];
        if NOT_A_PASSWORD_WORD.contains(&&**w) {
            continue;
        }
        return match rng.below(10) {
            0..=4 => {
                let mut c = w.chars();
                let first = c.next().map(|f| f.to_ascii_uppercase()).unwrap_or('X');
                format!("{first}{}", c.as_str())
            }
            5..=8 => w.to_string(),
            _ => w.to_ascii_uppercase(),
        };
    }
}

/// A word-based password: one or two dictionary words with digits and a symbol.
pub fn word_password(rng: &mut Rng, words: &WordSet) -> String {
    let digits = rand_var(rng, DIGITS, 1, 4);
    let sym = (SYMBOLS[rng.below(SYMBOLS.len())] as char).to_string();
    let (a, b) = (word(rng, words), word(rng, words));
    match rng.below(6) {
        0 => format!("{a}{digits}{sym}"),
        1 => format!("{a}{sym}{digits}"),
        2 => format!("{a}{b}{digits}{sym}"),
        3 => format!("{a}{sym}{b}{digits}"),
        4 => format!("{digits}{a}{sym}{b}"),
        _ => format!("{a}{digits}{sym}{b}"),
    }
}

/// A placeholder, a masked value or a reference for a secret under `var`.
fn placeholder(rng: &mut Rng, var: &str, real: &str) -> String {
    match rng.below(14) {
        0 => pick(
            rng,
            &[
                "your-api-key-here",
                "YOUR_API_KEY",
                "your_token_here",
                "INSERT_KEY_HERE",
            ],
        )
        .to_string(),
        1 => pick(
            rng,
            &["<your-token>", "<API_KEY>", "<your-password>", "<secret>"],
        )
        .to_string(),
        2 => "x".repeat(8 + rng.below(32)),
        3 => "*".repeat(6 + rng.below(20)),
        4 => pick(
            rng,
            &[
                "XXXX-XXXX-XXXX-XXXX",
                "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                "--------",
            ],
        )
        .to_string(),
        5 => pick(
            rng,
            &[
                "changeme",
                "CHANGE_ME",
                "replace-me",
                "dummy",
                "TODO",
                "placeholder",
                "password",
                "secret",
                "example",
            ],
        )
        .to_string(),
        6 => format!("${{{var}}}"),
        7 => format!("${var}"),
        8 => format!("{{{{ secrets.{var} }}}}"),
        9 => format!("%{var}%"),
        10 => pick(rng, &["process.env.", "os.environ.", "ENV.", "settings."]).to_string() + var,
        // redacted: the first characters, then stars
        11 => {
            let keep = real
                .find(['_', '-'])
                .map_or(4, |i| i + 1)
                .min(real.len() / 3);
            format!("{}{}", &real[..keep], "*".repeat(8 + rng.below(20)))
        }
        12 => format!("AKIA{}EXAMPLE", rand_str(rng, BASE32, 9)),
        _ => format!("{}...", &real[..real.len().min(6)]),
    }
}

fn uuid(rng: &mut Rng) -> String {
    format!(
        "{}-{}-4{}-{}{}-{}",
        rand_str(rng, HEX, 8),
        rand_str(rng, HEX, 4),
        rand_str(rng, HEX, 3),
        pick(rng, &["8", "9", "a", "b"]),
        rand_str(rng, HEX, 3),
        rand_str(rng, HEX, 12)
    )
}

/// A secret-looking value of no provider: hexadecimal or letters and digits.
fn random_secret(rng: &mut Rng) -> String {
    match rng.below(4) {
        0 => rand_str(rng, HEX, 32),
        1 => rand_str(rng, HEX, 64),
        2 => rand_var(rng, ALNUM, 24, 24),
        _ => rand_var(rng, B64, 32, 16),
    }
}

/// The candidates of one draw of `class`: one, or the two of a near-miss pair.
fn draw(rng: &mut Rng, class: &str, words: &WordSet) -> Vec<Option<Item>> {
    match class {
        "provider-key" => {
            let (text, var) = provider_secret(rng);
            let var = if rng.f64() < 0.3 && !text.contains("://") && !text.starts_with("-----") {
                pick(rng, &SECRET_VARS).to_string()
            } else {
                var
            };
            let cx = template_context(rng, &var);
            vec![candidate(&cx, &text, Label::T, class)]
        }
        "word-password" => {
            let var = pick(rng, &PASSWORD_VARS);
            let cx = template_context(rng, var);
            vec![candidate(&cx, &word_password(rng, words), Label::T, class)]
        }
        "near-miss" => {
            let (secret, var) = match rng.below(3) {
                0 => (
                    word_password(rng, words),
                    pick(rng, &PASSWORD_VARS).to_string(),
                ),
                1 => (random_secret(rng), pick(rng, &SECRET_VARS).to_string()),
                // a key that is its whole value: not a connection string, not a key block
                _ => loop {
                    let (text, var) = provider_secret(rng);
                    if !text.contains("://") && !text.starts_with("-----") {
                        break (text, var);
                    }
                },
            };
            let cx = template_context(rng, &var);
            let other = placeholder(rng, &var, &secret);
            vec![
                candidate(&cx, &secret, Label::T, "near-miss-secret"),
                candidate(&cx, &other, Label::X, "near-miss-placeholder"),
            ]
        }
        "test-key" => {
            let p = pick(rng, &["sk_test", "rk_test"]);
            let var = pick(
                rng,
                &[
                    "STRIPE_SECRET_KEY",
                    "STRIPE_API_KEY",
                    "API_KEY",
                    "PAYMENT_SECRET",
                ],
            );
            let cx = template_context(rng, var);
            vec![candidate(
                &cx,
                &format!("{p}_{}", stripe_body(rng)),
                Label::X,
                class,
            )]
        }
        "publishable-key" => {
            let p = pick(rng, &["pk_live", "pk_test"]);
            let var = pick(
                rng,
                &[
                    "STRIPE_PUBLISHABLE_KEY",
                    "NEXT_PUBLIC_STRIPE_KEY",
                    "PUBLIC_API_KEY",
                ],
            );
            let cx = template_context(rng, var);
            vec![candidate(
                &cx,
                &format!("{p}_{}", stripe_body(rng)),
                Label::F,
                class,
            )]
        }
        "placeholder" => {
            let (real, var) = provider_secret(rng);
            let var = if rng.f64() < 0.5 || real.contains("://") {
                pick(rng, &SECRET_VARS).to_string()
            } else {
                var
            };
            let cx = template_context(rng, &var);
            let v = placeholder(rng, &var, &real);
            vec![candidate(&cx, &v, Label::X, class)]
        }
        "public-key" => {
            let (var, value) = match rng.below(5) {
                0 => (
                    pick(rng, &["ssh_public_key", "deploy_key_pub", "authorized_key"]),
                    format!(
                        "ssh-rsa AAAAB3NzaC1yc2E{} deploy@build",
                        rand_var(rng, B64, 150, 60)
                    ),
                ),
                1 => (
                    pick(rng, &["ssh_public_key", "host_key", "signing_key_pub"]),
                    format!(
                        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI{}",
                        rand_str(rng, B64, 43)
                    ),
                ),
                2 => (
                    pick(rng, &["public_key", "jwt_public_key", "verify_key"]),
                    format!(
                        "-----BEGIN PUBLIC KEY-----\\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8A{}",
                        rand_var(rng, B64, 90, 60)
                    ),
                ),
                3 => (
                    pick(rng, &["tls_cert_key", "ca_cert_token", "auth_certificate"]),
                    format!(
                        "-----BEGIN CERTIFICATE-----\\nMIID{}",
                        rand_var(rng, B64, 110, 60)
                    ),
                ),
                _ => (
                    pick(rng, &["public_key", "verification_key", "api_public_key"]),
                    format!(
                        "MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE{}",
                        rand_str(rng, B64, 88)
                    ),
                ),
            };
            let cx = template_context(rng, &var.to_ascii_uppercase());
            vec![candidate(&cx, &value, Label::F, class)]
        }
        "checksum" => {
            let cx = match rng.below(8) {
                0 => lines_context(
                    pick(rng, &["assets/manifest.json", "vendor/modules.json", "public/build/manifest.json"]),
                    &["    \"version\": \"4.17.21\",", "    \"resolved\": \"https://registry.example.org/pkg/-/pkg-4.17.21.tgz\","],
                    format!("    \"integrity\": \"{SLOT}\","),
                    &["    \"dev\": true"],
                ),
                1 => lines_context(
                    pick(rng, &["public/index.html", "templates/base.html", "app/views/layout.html"]),
                    &["<head>"],
                    format!("<script src=\"https://cdn.example.org/lib.min.js\" integrity=\"{SLOT}\" crossorigin=\"anonymous\"></script>"),
                    &["</head>"],
                ),
                2 => lines_context(
                    pick(rng, &["build/deps.toml", "pkg/sources.toml", "flake.toml"]),
                    &["[source]", "url = \"https://example.org/archive/v1.2.3.tar.gz\""],
                    format!("sha256 = \"{SLOT}\""),
                    &[],
                ),
                3 => lines_context(
                    pick(rng, &["build/checksums.txt", "dist/SHA256SUMS", "release/checksums.yml"]),
                    &["artifacts:"],
                    format!("  checksum: sha256:{SLOT}"),
                    &["  size: 48213"],
                ),
                _ => {
                    let var = pick(rng, &["KEY_SHA256", "TOKEN_CHECKSUM", "API_HASH", "KEY_FINGERPRINT", "CACHE_KEY", "ETAG_KEY", "CHECKSUM"]);
                    template_context(rng, var)
                }
            };
            let v = if cx.lines[cx.at].contains("integrity") {
                match rng.below(3) {
                    0 => format!("sha512-{}==", rand_str(rng, B64, 86)),
                    1 => format!("sha384-{}", rand_str(rng, B64, 64)),
                    _ => format!("sha1-{}=", rand_str(rng, B64, 27)),
                }
            } else {
                match rng.below(5) {
                    0 => rand_str(rng, HEX, 40),
                    1 => rand_str(rng, HEX, 32),
                    2 => format!("SHA256:{}", rand_str(rng, B64, 43)),
                    3 => rand_str(rng, HEX, 128),
                    _ => rand_str(rng, HEX, 64),
                }
            };
            vec![candidate(&cx, &v, Label::F, class)]
        }
        "uuid" => {
            let var = pick(rng, &NEUTRAL_KEY_VARS);
            let cx = template_context(rng, var);
            vec![candidate(&cx, &uuid(rng), Label::F, class)]
        }
        "commit-id" => {
            let var = pick(
                rng,
                &[
                    "BUILD_KEY",
                    "COMMIT_TOKEN",
                    "GIT_COMMIT_KEY",
                    "RELEASE_KEY",
                    "SOURCE_KEY_REV",
                    "PIPELINE_TOKEN_SHA",
                ],
            );
            let cx = if rng.below(3) == 0 {
                lines_context(
                    pick(
                        rng,
                        &[
                            "deploy/values.yaml",
                            "k8s/deployment.yaml",
                            "docker-compose.yml",
                        ],
                    ),
                    &["    spec:", "      containers:"],
                    format!("        image_key: registry.example.org/app@sha256:{SLOT}"),
                    &["        ports:"],
                )
            } else {
                template_context(rng, var)
            };
            let v = if cx.lines[cx.at].contains("sha256:") {
                rand_str(rng, HEX, 64)
            } else {
                rand_str(rng, HEX, 40)
            };
            vec![candidate(&cx, &v, Label::F, class)]
        }
        "base64-data" => {
            let kind = pick(
                rng,
                &[
                    "image/png",
                    "image/gif",
                    "image/svg+xml",
                    "font/woff2",
                    "application/font-woff",
                ],
            );
            let body = match kind {
                "image/png" => format!("iVBORw0KGgoAAAANSUhEUgAA{}", rand_var(rng, B64, 80, 120)),
                "image/gif" => format!("R0lGODlh{}", rand_var(rng, B64, 60, 120)),
                "font/woff2" => format!("d09GMgABAAAAA{}", rand_var(rng, B64, 80, 120)),
                _ => rand_var(rng, B64, 80, 140),
            };
            let v = format!("data:{kind};base64,{body}");
            let cx = match rng.below(4) {
                0 => lines_context(
                    pick(
                        rng,
                        &[
                            "public/css/site.css",
                            "assets/styles/main.css",
                            "app/assets/theme.css",
                        ],
                    ),
                    &[".logo {", "  width: 32px;"],
                    format!("  --logo-key: \"{SLOT}\";"),
                    &["}"],
                ),
                1 => lines_context(
                    pick(rng, &["public/index.html", "templates/email.html"]),
                    &["<body>"],
                    format!("<img alt=\"key\" data-token=\"{SLOT}\">"),
                    &["</body>"],
                ),
                _ => {
                    let var = pick(
                        rng,
                        &[
                            "ICON_KEY",
                            "LOGO_TOKEN",
                            "FONT_KEY",
                            "AVATAR_TOKEN",
                            "FAVICON_KEY",
                        ],
                    );
                    template_context(rng, var)
                }
            };
            vec![candidate(&cx, &v, Label::F, class)]
        }
        "documentation" => {
            let var = pick(rng, &SECRET_VARS).to_string();
            let (real, _) = provider_secret(rng);
            let v = match rng.below(4) {
                0 => format!("AKIA{}EXAMPLE", rand_str(rng, BASE32, 9)),
                1 => format!("sk_test_{}", stripe_body(rng)),
                _ => placeholder(rng, &var, &real),
            };
            let file = docs_file(rng);
            let cx = match rng.below(5) {
                0 => lines_context(
                    &file,
                    &["Set your key before the first run:", "```sh"],
                    format!("export {var}={SLOT}"),
                    &["```"],
                ),
                1 => lines_context(
                    &file,
                    &["## Configuration", ""],
                    format!(
                        "Add `{}: {SLOT}` to `config.yml` (replace it with your own).",
                        lower(&var)
                    ),
                    &[""],
                ),
                2 => lines_context(
                    &file,
                    &["For example:", "```"],
                    format!(
                        "curl -H \"Authorization: Bearer {SLOT}\" https://api.example.com/v1/items"
                    ),
                    &["```"],
                ),
                3 => lines_context(
                    &file,
                    &["Options:", ""],
                    format!("- `{}`: `{SLOT}` (an example value)", camel(&var)),
                    &["- `timeout`: `30`"],
                ),
                _ => lines_context(
                    &file,
                    &["```python", "# example only"],
                    format!("{} = \"{SLOT}\"", lower(&var)),
                    &["client = Client(api_key)", "```"],
                ),
            };
            vec![candidate(&cx, &v, Label::X, class)]
        }
        // a secret-looking value in a test file, under a name that says so
        _ => {
            let var = pick(
                rng,
                &[
                    "FAKE_TOKEN",
                    "TEST_API_KEY",
                    "DUMMY_SECRET",
                    "MOCK_PASSWORD",
                    "TEST_TOKEN",
                    "FIXTURE_KEY",
                    "STUB_API_SECRET",
                    "SAMPLE_PASSWORD",
                ],
            );
            let mut cx = template_context(rng, var);
            cx.file = test_file(rng, &cx.file);
            let v = match rng.below(4) {
                0 => provider_secret(rng).0,
                1 => word_password(rng, words),
                _ => random_secret(rng),
            };
            vec![candidate(&cx, &v, Label::X, "test-fixture")]
        }
    }
}

fn draw_class(rng: &mut Rng) -> &'static str {
    let total: usize = CLASSES.iter().map(|c| c.1).sum();
    let mut x = rng.below(total);
    for (name, w) in CLASSES {
        if x < w {
            return name;
        }
        x -= w;
    }
    CLASSES[0].0
}

/// What makes two synthetic candidates the same one.
fn identity(it: &Item) -> u64 {
    let r = &it.rec;
    hash64(
        format!(
            "{}\u{1}{}\u{1}{}\u{1}{:?}\u{1}{:?}\u{1}{:?}\u{1}{}",
            r.value,
            r.line_text,
            r.start,
            r.prev_line,
            r.next_line,
            r.file,
            it.label.as_str()
        )
        .as_bytes(),
    )
}

/// `n_train` training and `n_val` validation candidates, all different: a validation candidate
/// is never a training one. The draws depend on the seed alone.
pub fn generate(
    seed: u64,
    n_train: usize,
    n_val: usize,
    words: &WordSet,
) -> Result<(Vec<Item>, Vec<Item>)> {
    let mut seen: HashSet<u64> = HashSet::new();
    let mut out = Vec::new();
    for (split, n) in [("train", n_train), ("val", n_val)] {
        let mut rng = Rng::derived(seed, &format!("s6-secrets2:synth:{split}"));
        let mut items: Vec<Item> = Vec::with_capacity(n);
        let mut tries = 0usize;
        while items.len() < n {
            tries += 1;
            if tries > 50 * n + 10_000 {
                anyhow::bail!("the synthetic generator cannot draw {n} different candidates");
            }
            let class = draw_class(&mut rng);
            let drawn: Vec<Item> = draw(&mut rng, class, words).into_iter().flatten().collect();
            // a pair goes in whole or not at all
            let room = n - items.len();
            if drawn.is_empty() || drawn.len() > room {
                continue;
            }
            let ids: Vec<u64> = drawn.iter().map(identity).collect();
            if ids.iter().any(|h| seen.contains(h)) || (ids.len() == 2 && ids[0] == ids[1]) {
                continue;
            }
            seen.extend(ids);
            for mut it in drawn {
                it.id = format!("synth:{split}:{}", items.len());
                it.split = split.to_string();
                items.push(it);
            }
        }
        out.push(items);
    }
    let val = out.pop().unwrap_or_default();
    let train = out.pop().unwrap_or_default();
    Ok((train, val))
}

/// The number of synthetic training candidates: [`TRAIN_TARGET`], or fewer when a quarter of
/// the training candidates is fewer (section 3.3: at most 25% of training items, so at most a
/// third of the candidates of the two other sources).
pub fn train_count(creddata_train: usize, issues_train: usize) -> usize {
    TRAIN_TARGET.min((creddata_train + issues_train) / 3)
}

/// Writes `<out>/candidates/synth-{train,val}.jsonl` and `<out>/counts/synth.json`. Without
/// `n_train`, the count comes from the two other sources' counts in `<out>/counts`
/// ([`train_count`]).
pub fn write(
    out: &std::path::Path,
    seed: u64,
    n_train: Option<usize>,
    n_val: usize,
    words: &WordSet,
) -> Result<serde_json::Value> {
    use serde_json::json;
    let (n_train, rule) = match n_train {
        Some(n) => (n, json!("given")),
        None => {
            let cred = super::read_json(&out.join("counts/creddata.json"))?;
            let issues = super::read_json(&out.join("counts/issues.json"))?;
            let c = cred["items"]["training"]["kept"]["items"].as_u64();
            let i = issues["splits"]["train"]["candidates"].as_u64();
            let (Some(c), Some(i)) = (c, i) else {
                anyhow::bail!("counts/creddata.json and counts/issues.json do not give the training candidates");
            };
            let n = train_count(c as usize, i as usize);
            (
                n,
                json!({
                    "creddata_training_candidates": c,
                    "issue_report_training_candidates": i,
                    "cap_a_quarter_of_training_items": (c + i) / 3,
                    "target": TRAIN_TARGET,
                }),
            )
        }
    };
    let (train, val) = generate(seed, n_train, n_val, words)?;
    let mut splits = serde_json::Map::new();
    for (split, items) in [("train", &train), ("val", &val)] {
        crate::record::write_jsonl(
            &out.join(format!("candidates/synth-{split}.jsonl")),
            items.iter(),
        )?;
        let mut by_class: std::collections::BTreeMap<String, super::Counts> = Default::default();
        let mut by_label = super::Counts::new();
        for it in items {
            super::bump(&mut by_label, it.label.as_str());
            super::bump(
                by_class
                    .entry(it.class.clone().unwrap_or_default())
                    .or_default(),
                it.label.as_str(),
            );
        }
        splits.insert(
            split.to_string(),
            json!({"candidates": items.len(), "by_label": by_label, "by_class": by_class}),
        );
    }
    let counts = json!({
        "generator": GENERATOR,
        "training_candidates": n_train,
        "training_count_rule": rule,
        "validation_candidates": n_val,
        "splits": splits,
    });
    super::write_json(&out.join("counts/synth.json"), &counts)?;
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn the_synthetic_share_is_capped_at_a_quarter() {
        assert_eq!(train_count(25_000, 40_000), TRAIN_TARGET);
        assert_eq!(train_count(9_000, 30_000), 13_000);
        // 13,000 of 52,000 training candidates: a quarter
        assert_eq!(13_000 * 4, 9_000 + 30_000 + 13_000);
        assert_eq!(train_count(0, 0), 0);
    }

    fn words() -> WordSet {
        WordSet::from_text(
            "tree\nbird\nlamp\nhouse\nhorse\nmouse\nwinter\nsummer\nspring\nbattery\ncabinet\nmountain\nsandwich\npassword\nsecret\n",
        )
    }

    fn sample() -> (Vec<Item>, Vec<Item>) {
        generate(1234, 3000, 400, &words()).unwrap()
    }

    #[test]
    fn the_draws_are_deterministic_and_the_splits_disjoint() {
        let (a, av) = sample();
        let (b, bv) = sample();
        assert_eq!((a.len(), av.len()), (3000, 400));
        assert!(a == b && av == bv);
        let (c, _) = generate(1235, 300, 0, &words()).unwrap();
        assert_ne!(a[..300], c[..]);
        let train: HashSet<u64> = a.iter().map(identity).collect();
        assert_eq!(train.len(), a.len());
        assert!(av.iter().all(|i| !train.contains(&identity(i))));
        assert!(a
            .iter()
            .enumerate()
            .all(|(i, it)| it.id == format!("synth:train:{i}") && it.split == "train"));
        assert!(av.iter().all(|it| it.split == "val" && it.kind && !it.weak));
    }

    #[test]
    fn records_hold_their_value_where_they_say() {
        let (items, _) = sample();
        for it in &items {
            let r = &it.rec;
            assert_eq!(r.generator, GENERATOR);
            assert!(!r.value.is_empty(), "{:?}", it.class);
            // the first line of the value is in the line, at its columns (ASCII lines)
            let first = r.value.split('\n').next().unwrap();
            assert_eq!(
                &r.line_text[r.start - r.line_from..][..first.len()],
                first,
                "{:?}",
                it.class
            );
            assert_eq!(r.length, r.value.len());
            assert!(r.file.is_some());
            assert_eq!(r.filters, flags_of(&r.rule, &r.value));
            assert!(!r.filters.stopword && !r.filters.allowlist && !r.filters.context);
        }
    }

    #[test]
    fn every_class_is_drawn_with_its_label() {
        let (items, _) = sample();
        let mut by: BTreeMap<String, BTreeMap<&str, usize>> = BTreeMap::new();
        for it in &items {
            *by.entry(it.class.clone().unwrap())
                .or_default()
                .entry(it.label.as_str())
                .or_default() += 1;
        }
        let label_of = |class: &str| {
            let m = &by[class];
            assert_eq!(m.len(), 1, "{class}: {m:?}");
            *m.keys().next().unwrap()
        };
        for class in ["provider-key", "word-password", "near-miss-secret"] {
            assert_eq!(label_of(class), "T");
        }
        for class in [
            "test-key",
            "placeholder",
            "documentation",
            "test-fixture",
            "near-miss-placeholder",
        ] {
            assert_eq!(label_of(class), "X");
        }
        for class in [
            "publishable-key",
            "public-key",
            "checksum",
            "uuid",
            "commit-id",
            "base64-data",
        ] {
            assert_eq!(label_of(class), "F");
        }
        assert_eq!(by.len(), 14);
        let t = items.iter().filter(|i| i.label == Label::T).count();
        assert!((1050..1350).contains(&t), "{t} positives of 3000");
        // test fixtures sit under a test scope, documentation under none
        for it in &items {
            let scope = sky_infer::secret_cand::path_scope(it.rec.file.as_deref().unwrap()).0;
            match it.class.as_deref() {
                Some("test-fixture") => assert!(
                    scope.contains(&"test")
                        || scope.contains(&"spec")
                        || scope.contains(&"fixture")
                        || scope.contains(&"mock"),
                    "{:?}",
                    it.rec.file
                ),
                Some("provider-key" | "word-password") => {
                    assert!(!scope.contains(&"test"), "{:?}", it.rec.file)
                }
                _ => {}
            }
        }
    }

    #[test]
    fn near_miss_pairs_share_their_lines() {
        let (items, _) = sample();
        let mut pairs = 0;
        for w in items.windows(2) {
            if w[0].class.as_deref() != Some("near-miss-secret") {
                continue;
            }
            assert_eq!(w[1].class.as_deref(), Some("near-miss-placeholder"));
            let (a, b) = (&w[0].rec, &w[1].rec);
            assert_eq!(
                (
                    a.file.as_ref(),
                    a.line,
                    a.prev_line.as_ref(),
                    a.next_line.as_ref()
                ),
                (
                    b.file.as_ref(),
                    b.line,
                    b.prev_line.as_ref(),
                    b.next_line.as_ref()
                )
            );
            // the same line around two values
            if a.start == b.start && a.end_line == a.line {
                let around = |r: &super::super::CandRecord| {
                    format!(
                        "{}\u{1}{}",
                        &r.line_text[..r.start],
                        &r.line_text[r.start + r.value.len()..]
                    )
                };
                assert_eq!(around(a), around(b));
            }
            assert_ne!(a.value, b.value);
            pairs += 1;
        }
        assert!(pairs > 60, "{pairs}");
    }

    #[test]
    fn aws_ids_use_the_alphabet_aws_uses() {
        let mut rng = Rng::derived(1, "aws");
        for _ in 0..200 {
            let id = aws_access_key_id(&mut rng);
            assert_eq!(id.len(), 20);
            assert!(
                id[4..].bytes().all(|b| BASE32.contains(&b)),
                "a digit AWS never issues"
            );
        }
        let (items, _) = sample();
        let ids: Vec<&str> = items
            .iter()
            .map(|i| i.rec.value.as_str())
            .filter(|v| v.starts_with("AKIA") && !v.ends_with("EXAMPLE") && v.len() == 20)
            .collect();
        assert!(ids.len() > 10);
        assert!(ids
            .iter()
            .all(|v| v[4..].bytes().all(|b| BASE32.contains(&b))));
    }

    #[test]
    fn word_passwords_hold_a_word_digits_and_a_symbol() {
        let w = words();
        let mut rng = Rng::derived(4, "words");
        let mut with_run = 0;
        for _ in 0..300 {
            let p = word_password(&mut rng, &w);
            // two words written together are one run of letters, which is no dictionary word:
            // the word-run feature reads the others
            with_run += (sky_infer::secret_cand::longest_word_run(&p, &w) >= 4) as usize;
            assert!(p.bytes().filter(u8::is_ascii_alphabetic).count() >= 4);
            assert!(
                p.bytes().any(|b| b.is_ascii_digit()) && p.bytes().any(|b| SYMBOLS.contains(&b))
            );
            let low = p.to_ascii_lowercase();
            assert!(!low.contains("password") && !low.contains("secret"));
        }
        assert!((200..300).contains(&with_run), "{with_run}");
    }

    #[test]
    fn rules_and_flags() {
        let mut rng = Rng::derived(6, "rules");
        let hex = rand_str(&mut rng, HEX, 40);
        assert_eq!(rule_of(&hex, Some("API_KEY"), false), "generic-api-key");
        assert_eq!(rule_of(&hex, Some("BUILD_ID"), false), "generic-keyword");
        assert_eq!(
            rule_of("Tree42!x", Some("db_password"), true),
            "password-assignment"
        );
        assert_eq!(
            rule_of("Tree42!x", Some("db_password"), false),
            "generic-keyword"
        );
        assert_eq!(
            rule_of(&aws_access_key_id(&mut rng), Some("X"), false),
            "aws-access-token"
        );
        let pem = format!("-----BEGIN {} KEY-----", "PUBLIC");
        assert_eq!(rule_of(&pem, Some("public_key"), true), "generic-keyword");
        assert!(flags_of("generic-api-key", "aaaaaaaaaaaaaaaa").entropy);
        assert!(!flags_of("generic-api-key", &hex).entropy);
        assert!(!flags_of("generic-keyword", "aaaa").entropy);
        for v in [
            "xxxxxxxx",
            "********",
            "${API_KEY}",
            "$TOKEN",
            "<your-token>",
            "{{ secrets.KEY }}",
            "%KEY%",
            "changeme",
            "abab",
            "sk-ab****",
        ] {
            assert!(looks_like_placeholder(v), "{v}");
        }
        let example = format!("AKIA{}EXAMPLE", rand_str(&mut rng, BASE32, 9));
        assert!(looks_like_placeholder(&example));
        assert!(!looks_like_placeholder(&hex) && !looks_like_placeholder("Tree42!x"));
        assert_eq!(
            name_before("  db_password = \"").as_deref(),
            Some("db_password")
        );
        assert_eq!(
            name_before("  -H \"Authorization: Bearer ").as_deref(),
            Some("Authorization")
        );
        assert_eq!(name_before("const apiKey = '").as_deref(), Some("apiKey"));
        assert_eq!(
            name_before("      - STRIPE_KEY=").as_deref(),
            Some("STRIPE_KEY")
        );
        assert_eq!(
            name_before("\tclientSecret := \"").as_deref(),
            Some("clientSecret")
        );
        assert_eq!(name_before("  \"token\": \"").as_deref(), Some("token"));
        assert_eq!(name_before("").as_deref(), None);
    }

    #[test]
    fn a_connection_string_gives_its_password() {
        let (items, _) = sample();
        let urls: Vec<&Item> = items
            .iter()
            .filter(|i| i.rec.rule == "url-credentials")
            .collect();
        assert!(!urls.is_empty());
        for it in urls {
            let r = &it.rec;
            assert!(
                r.line_text[..r.start].ends_with(':')
                    && r.line_text[r.start + r.value.len()..].starts_with('@')
            );
            assert!(!r.value.contains('@') && !r.value.contains("://"));
        }
    }
}
