//! API-key and secret detection as typed decisions (issue #59): short texts (an `.env` file, a
//! code snippet, a config, a `curl` command, a log, a diff) that may expose a credential, with
//! four questions per text: is a usable secret exposed (noul), which service issued it, what
//! kind of value it is, and on which line.
//!
//! Everything is generated locally and deterministically from the seed. Keys follow each
//! provider's public format with random bodies, so none is a real key; they are written to
//! `data/secrets` only and never committed. The negatives are the hard ones a scanner has to
//! tell apart: test keys, publishable keys, placeholders, redacted values, environment
//! references, and random-looking values that are not credentials (hashes, UUIDs, commit ids),
//! often under a variable name that sounds secret.
//!
//! `secret_synth` (train / val by text) draws from [`TRAIN_TEMPLATES`]; `secret_bench` (test)
//! draws only from [`BENCH_TEMPLATES`], contexts the training set never shows. [`baseline`] is
//! the rule scanner the model is compared with (gitleaks-style prefixes plus an entropy rule on
//! secret-sounding names).

use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::decisions::{finalize, DecisionOptions, Item, SourceReport};
use crate::record::{one_hot, Question};
use crate::rng::Rng;
use crate::typed::state_group;

const UPPER_DIGITS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
const ALNUM: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
const ALPHA: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const URLSAFE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
const B64: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const HEX: &[u8] = b"0123456789abcdef";
const DIGITS: &[u8] = b"0123456789";

fn rand_str(rng: &mut Rng, set: &[u8], n: usize) -> String {
    (0..n).map(|_| set[rng.below(set.len())] as char).collect()
}

/// `min + below(spread)` characters of `set`.
fn rand_var(rng: &mut Rng, set: &[u8], min: usize, spread: usize) -> String {
    let n = min + rng.below(spread);
    rand_str(rng, set, n)
}

/// A Stripe key body: 24, 61 or 98 characters.
fn stripe_body(rng: &mut Rng) -> String {
    let n = 24 + rng.below(3) * 37;
    rand_str(rng, ALNUM, n)
}

fn pick<'a, T: ?Sized>(rng: &mut Rng, xs: &[&'a T]) -> &'a T {
    xs[rng.below(xs.len())]
}

/// The services of the "which service" question, with the options' definitions (empty: the
/// key alone). `none` comes last.
pub const SERVICES: [(&str, &str); 15] = [
    ("aws", ""),
    ("github", ""),
    ("gitlab", ""),
    ("stripe", ""),
    ("slack", ""),
    ("openai", ""),
    ("anthropic", ""),
    ("google", ""),
    ("huggingface", ""),
    ("sendgrid", ""),
    ("npm", ""),
    ("private_key", "a PEM private key"),
    ("database", "a connection string with its password"),
    ("generic", "a key or token of another service"),
    ("none", "no actual key or token"),
];

/// The kinds of the "what kind of value" question.
pub const KINDS: [(&str, &str); 5] = [
    ("secret", "a live, usable credential"),
    ("test", "a test or sandbox credential"),
    ("public", "a publishable key meant to be public"),
    (
        "placeholder",
        "a placeholder, an example, a redacted value or a reference",
    ),
    ("other", "not a credential, even if it looks random"),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Secret,
    Test,
    Public,
    Placeholder,
    Other,
}

impl Kind {
    fn index(self) -> usize {
        self as usize
    }
}

/// One generated value: the text placed in the template, the variable name it is assigned to,
/// its kind and service (index in [`SERVICES`]).
#[derive(Clone, Debug)]
pub struct Value {
    pub text: String,
    pub var: String,
    pub kind: Kind,
    pub service: usize,
}

fn service(key: &str) -> usize {
    SERVICES.iter().position(|(k, _)| *k == key).unwrap()
}

const NONE: usize = SERVICES.len() - 1;

/// Variable names that name no service: the value's format must carry the label.
const NEUTRAL_VARS: [&str; 8] = [
    "TOKEN",
    "API_KEY",
    "AUTH",
    "SECRET",
    "ACCESS_TOKEN",
    "CREDENTIALS",
    "KEY",
    "PASSWORD",
];

/// Names of non-credential values; some sound secret on purpose.
const OTHER_VARS: [&str; 10] = [
    "BUILD_SHA",
    "COMMIT",
    "REQUEST_ID",
    "TRACE_ID",
    "CHECKSUM",
    "CACHE_KEY",
    "DEVICE_ID",
    "ETAG",
    "KEY_FINGERPRINT",
    "SESSION_ID",
];

/// A live secret of `svc` and the provider's variable name.
fn live(rng: &mut Rng, svc: &str) -> (String, &'static str) {
    match svc {
        "aws" => {
            if rng.f64() < 0.5 {
                (
                    format!("AKIA{}", rand_str(rng, UPPER_DIGITS, 16)),
                    "AWS_ACCESS_KEY_ID",
                )
            } else {
                (rand_str(rng, B64, 40), "AWS_SECRET_ACCESS_KEY")
            }
        }
        "github" => {
            if rng.f64() < 0.25 {
                (
                    format!(
                        "github_pat_{}_{}",
                        rand_str(rng, ALNUM, 22),
                        rand_str(rng, ALNUM, 59)
                    ),
                    "GITHUB_TOKEN",
                )
            } else {
                let p = pick(rng, &["ghp", "gho", "ghu", "ghs"]);
                (
                    format!("{p}_{}", rand_str(rng, ALNUM, 36)),
                    pick(rng, &["GITHUB_TOKEN", "GH_TOKEN"]),
                )
            }
        }
        "gitlab" => (
            format!("glpat-{}", rand_str(rng, URLSAFE, 20)),
            "GITLAB_TOKEN",
        ),
        "stripe" => {
            let p = pick(rng, &["sk_live", "rk_live"]);
            (format!("{p}_{}", stripe_body(rng)), "STRIPE_SECRET_KEY")
        }
        "slack" => {
            if rng.f64() < 0.3 {
                (
                    format!(
                        "https://hooks.slack.com/services/T{}/B{}/{}",
                        rand_str(rng, UPPER_DIGITS, 8),
                        rand_str(rng, UPPER_DIGITS, 10),
                        rand_str(rng, ALNUM, 24)
                    ),
                    "SLACK_WEBHOOK_URL",
                )
            } else {
                let p = pick(rng, &["xoxb", "xoxp"]);
                (
                    format!(
                        "{p}-{}-{}-{}",
                        rand_str(rng, DIGITS, 11),
                        rand_str(rng, DIGITS, 13),
                        rand_str(rng, ALNUM, 24)
                    ),
                    "SLACK_BOT_TOKEN",
                )
            }
        }
        "openai" => {
            if rng.f64() < 0.6 {
                (
                    format!("sk-proj-{}", rand_str(rng, URLSAFE, 64)),
                    "OPENAI_API_KEY",
                )
            } else {
                (format!("sk-{}", rand_str(rng, ALNUM, 48)), "OPENAI_API_KEY")
            }
        }
        "anthropic" => (
            format!("sk-ant-api03-{}AA", rand_str(rng, URLSAFE, 93)),
            "ANTHROPIC_API_KEY",
        ),
        "google" => (
            format!("AIza{}", rand_str(rng, URLSAFE, 35)),
            pick(rng, &["GOOGLE_API_KEY", "MAPS_API_KEY", "FIREBASE_API_KEY"]),
        ),
        "huggingface" => (format!("hf_{}", rand_str(rng, ALPHA, 34)), "HF_TOKEN"),
        "sendgrid" => (
            format!(
                "SG.{}.{}",
                rand_str(rng, URLSAFE, 22),
                rand_str(rng, URLSAFE, 43)
            ),
            "SENDGRID_API_KEY",
        ),
        "npm" => (format!("npm_{}", rand_str(rng, ALNUM, 36)), "NPM_TOKEN"),
        "private_key" => {
            let t = pick(rng, &["RSA ", "EC ", "OPENSSH ", ""]);
            (
                format!(
                    "-----BEGIN {t}PRIVATE KEY-----\\n{}\\n{}\\n-----END {t}PRIVATE KEY-----",
                    rand_str(rng, B64, 64),
                    rand_var(rng, B64, 16, 24)
                ),
                pick(rng, &["PRIVATE_KEY", "SSH_KEY", "JWT_SIGNING_KEY"]),
            )
        }
        "database" => {
            let scheme = pick(rng, &["postgres", "mysql", "mongodb", "redis"]);
            let user = pick(rng, &["app", "admin", "billing", "svc_api"]);
            let host = pick(rng, &["db.internal", "10.0.3.17", "prod-db-1", "cache"]);
            (
                format!(
                    "{scheme}://{user}:{}@{host}:5432/main",
                    rand_var(rng, ALNUM, 12, 12)
                ),
                pick(rng, &["DATABASE_URL", "DB_URL", "REDIS_URL"]),
            )
        }
        _ => {
            let body = if rng.f64() < 0.5 {
                rand_str(rng, HEX, 32)
            } else {
                rand_str(rng, ALNUM, 40)
            };
            (
                body,
                pick(
                    rng,
                    &[
                        "MAILGUN_API_KEY",
                        "API_SECRET",
                        "X_API_KEY",
                        "SERVICE_TOKEN",
                        "DATADOG_API_KEY",
                        "CLIENT_SECRET",
                    ],
                ),
            )
        }
    }
}

/// The services that issue live secrets (all but `none`).
const LIVE_SERVICES: usize = SERVICES.len() - 1;

/// A value of the given kind.
pub fn value(rng: &mut Rng, kind: Kind) -> Value {
    let neutral = rng.f64() < 0.35;
    let (text, var, service) = match kind {
        Kind::Secret => {
            let s = rng.below(LIVE_SERVICES);
            let (text, var) = live(rng, SERVICES[s].0);
            // a generic secret has nothing but its name: keep it
            let var = if neutral && SERVICES[s].0 != "generic" {
                pick(rng, &NEUTRAL_VARS).to_string()
            } else {
                var.to_string()
            };
            (text, var, s)
        }
        Kind::Test => {
            let p = pick(rng, &["sk_test", "rk_test"]);
            (
                format!("{p}_{}", stripe_body(rng)),
                if neutral {
                    "API_KEY"
                } else {
                    "STRIPE_SECRET_KEY"
                }
                .to_string(),
                service("stripe"),
            )
        }
        Kind::Public => {
            let p = pick(rng, &["pk_live", "pk_test"]);
            (
                format!("{p}_{}", stripe_body(rng)),
                pick(rng, &["STRIPE_PUBLISHABLE_KEY", "NEXT_PUBLIC_STRIPE_KEY"]).to_string(),
                service("stripe"),
            )
        }
        Kind::Placeholder => {
            let s = rng.below(LIVE_SERVICES);
            let (real, var) = live(rng, SERVICES[s].0);
            let var = if neutral {
                pick(rng, &NEUTRAL_VARS)
            } else {
                var
            };
            let text = match rng.below(8) {
                0 => pick(rng, &["your-api-key-here", "YOUR_API_KEY", "<your-token>"]).to_string(),
                1 => "x".repeat(16 + rng.below(24)),
                2 => pick(rng, &["changeme", "TODO", "replace-me", "dummy"]).to_string(),
                3 => format!("${{{var}}}"),
                4 => format!("{{{{ secrets.{var} }}}}"),
                // redacted: the provider's prefix, then stars
                5 => {
                    let keep = real
                        .find(['_', '-'])
                        .map_or(4, |i| i + 1)
                        .min(real.len() / 3);
                    format!("{}{}", &real[..keep], "*".repeat(12 + rng.below(20)))
                }
                6 => format!("AKIA{}EXAMPLE", rand_str(rng, UPPER_DIGITS, 9)),
                _ => format!("{}...", &real[..real.len().min(6)]),
            };
            (text, var.to_string(), NONE)
        }
        Kind::Other => {
            let var = pick(rng, &OTHER_VARS).to_string();
            let text = match rng.below(6) {
                0 => rand_str(rng, HEX, 40),
                1 => rand_str(rng, HEX, 64),
                2 => rand_str(rng, HEX, 32),
                3 => format!(
                    "{}-{}-4{}-{}{}-{}",
                    rand_str(rng, HEX, 8),
                    rand_str(rng, HEX, 4),
                    rand_str(rng, HEX, 3),
                    pick(rng, &["8", "9", "a", "b"]),
                    rand_str(rng, HEX, 3),
                    rand_str(rng, HEX, 12)
                ),
                4 => format!("SHA256:{}", rand_str(rng, B64, 43)),
                _ => rand_str(rng, DIGITS, 18),
            };
            (text, var, NONE)
        }
    };
    Value {
        text,
        var,
        kind,
        service,
    }
}

/// `SNAKE_CASE` → `snake_case` and `camelCase`.
fn lower(var: &str) -> String {
    var.to_ascii_lowercase()
}

fn camel(var: &str) -> String {
    let mut out = String::new();
    for (i, part) in var.split('_').filter(|p| !p.is_empty()).enumerate() {
        let p = part.to_ascii_lowercase();
        if i == 0 {
            out.push_str(&p);
        } else {
            let mut c = p.chars();
            if let Some(f) = c.next() {
                out.push(f.to_ascii_uppercase());
                out.push_str(c.as_str());
            }
        }
    }
    out
}

/// A context: lines around the value, and the index of the value's line.
type Template = fn(&mut Rng, &Value) -> (Vec<String>, usize);

fn with_filler(
    rng: &mut Rng,
    before: &[&str],
    line: String,
    after: &[&str],
) -> (Vec<String>, usize) {
    let nb = rng.below(before.len() + 1);
    let na = rng.below(after.len() + 1);
    let mut out: Vec<String> = before[before.len() - nb..]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let at = out.len();
    out.push(line);
    out.extend(after[..na].iter().map(|s| s.to_string()));
    (out, at)
}

fn t_env(rng: &mut Rng, v: &Value) -> (Vec<String>, usize) {
    with_filler(
        rng,
        &["# production", "NODE_ENV=production", "PORT=8080"],
        format!("{}={}", v.var, v.text),
        &["LOG_LEVEL=info", "REGION=eu-west-1", "WORKERS=4"],
    )
}

fn t_export(rng: &mut Rng, v: &Value) -> (Vec<String>, usize) {
    with_filler(
        rng,
        &["#!/usr/bin/env bash", "set -euo pipefail"],
        format!("export {}=\"{}\"", v.var, v.text),
        &["./bin/deploy --stage prod", "echo done"],
    )
}

fn t_python(rng: &mut Rng, v: &Value) -> (Vec<String>, usize) {
    let name = lower(&v.var);
    let use_line = format!("client = Client(api_key={name}, timeout=30)");
    with_filler(
        rng,
        &["import os", "from service import Client"],
        format!("{name} = \"{}\"", v.text),
        &[use_line.as_str(), "print(client.status())"],
    )
}

fn t_js(rng: &mut Rng, v: &Value) -> (Vec<String>, usize) {
    let name = camel(&v.var);
    let use_line = format!("const client = createClient({{ auth: {name} }});");
    with_filler(
        rng,
        &["import { createClient } from './api';"],
        format!("const {name} = '{}';", v.text),
        &[use_line.as_str(), "export default client;"],
    )
}

fn t_yaml(rng: &mut Rng, v: &Value) -> (Vec<String>, usize) {
    let (mut lines, at) = with_filler(
        rng,
        &["  name: billing", "  replicas: 2"],
        format!("  {}: {}", lower(&v.var), v.text),
        &["  region: us-east-1", "  retries: 3"],
    );
    lines.insert(0, "service:".into());
    (lines, at + 1)
}

fn t_json(rng: &mut Rng, v: &Value) -> (Vec<String>, usize) {
    let (mut lines, at) = with_filler(
        rng,
        &["  \"name\": \"worker\","],
        format!("  \"{}\": \"{}\",", camel(&v.var), v.text),
        &["  \"retries\": 3,", "  \"debug\": false"],
    );
    lines.insert(0, "{".into());
    lines.push("}".into());
    (lines, at + 1)
}

fn t_curl(rng: &mut Rng, v: &Value) -> (Vec<String>, usize) {
    let header = if rng.f64() < 0.5 {
        format!("  -H \"Authorization: Bearer {}\" \\", v.text)
    } else {
        format!("  -H \"X-Api-Key: {}\" \\", v.text)
    };
    let first = pick(
        rng,
        &[
            "curl -s https://api.example.com/v1/items \\",
            "curl -X POST https://api.example.com/v2/jobs \\",
        ],
    );
    let mut lines = vec![first.to_string(), header];
    lines.push("  -H \"Content-Type: application/json\"".into());
    (lines, 1)
}

fn t_log(rng: &mut Rng, v: &Value) -> (Vec<String>, usize) {
    with_filler(
        rng,
        &[
            "2026-10-12T08:14:03Z INFO request started path=/v1/sync",
            "2026-10-12T08:14:03Z DEBUG loading config from env",
        ],
        format!(
            "2026-10-12T08:14:04Z DEBUG auth configured {}={}",
            lower(&v.var),
            v.text
        ),
        &[
            "2026-10-12T08:14:04Z INFO upstream status=200 ms=41",
            "2026-10-12T08:14:05Z WARN retrying in 2s",
        ],
    )
}

fn t_diff(rng: &mut Rng, v: &Value) -> (Vec<String>, usize) {
    let (mut lines, at) = with_filler(
        rng,
        &[" def connect():", "     \"\"\"Open the session.\"\"\""],
        format!("+    {} = \"{}\"", lower(&v.var), v.text),
        &["     return Session(timeout=10)", "-    return None"],
    );
    lines.insert(0, "@@ -12,6 +12,7 @@".into());
    (lines, at + 1)
}

fn t_compose(rng: &mut Rng, v: &Value) -> (Vec<String>, usize) {
    let (mut lines, at) = with_filler(
        rng,
        &["      - NODE_ENV=production"],
        format!("      - {}={}", v.var, v.text),
        &["      - PORT=3000"],
    );
    let head = [
        "services:",
        "  api:",
        "    image: acme/api:1.4",
        "    environment:",
    ];
    for (i, h) in head.iter().enumerate() {
        lines.insert(i, h.to_string());
    }
    (lines, at + head.len())
}

fn t_toml(rng: &mut Rng, v: &Value) -> (Vec<String>, usize) {
    let (mut lines, at) = with_filler(
        rng,
        &["endpoint = \"https://api.example.com\""],
        format!("{} = \"{}\"", lower(&v.var), v.text),
        &["timeout = 30", "verbose = false"],
    );
    lines.insert(0, "[api]".into());
    (lines, at + 1)
}

fn t_go(rng: &mut Rng, v: &Value) -> (Vec<String>, usize) {
    let name = camel(&v.var);
    let use_line = format!("\tclient := api.New({name})");
    let (mut lines, at) = with_filler(
        rng,
        &["func main() {"],
        format!("\t{name} := \"{}\"", v.text),
        &[use_line.as_str(), "\tlog.Fatal(client.Run())"],
    );
    if lines[0] != "func main() {" {
        lines.insert(0, "func main() {".into());
        return (lines, at + 1);
    }
    (lines, at)
}

fn t_ini(rng: &mut Rng, v: &Value) -> (Vec<String>, usize) {
    let (mut lines, at) = with_filler(
        rng,
        &["host = smtp.example.com", "port = 587"],
        format!("{} = {}", lower(&v.var), v.text),
        &["use_tls = true"],
    );
    lines.insert(0, "[mailer]".into());
    (lines, at + 1)
}

/// The training contexts.
pub const TRAIN_TEMPLATES: [(&str, Template); 9] = [
    ("env", t_env),
    ("export", t_export),
    ("python", t_python),
    ("js", t_js),
    ("yaml", t_yaml),
    ("json", t_json),
    ("curl", t_curl),
    ("log", t_log),
    ("diff", t_diff),
];

/// The contexts of `secret_bench` only: the bench measures detection in files the training
/// never showed.
pub const BENCH_TEMPLATES: [(&str, Template); 4] = [
    ("compose", t_compose),
    ("toml", t_toml),
    ("go", t_go),
    ("ini", t_ini),
];

/// The shares of the kinds: half the texts expose a secret, so the noul question is balanced.
const KIND_SHARES: [(Kind, f64); 5] = [
    (Kind::Secret, 0.5),
    (Kind::Test, 0.1),
    (Kind::Public, 0.08),
    (Kind::Placeholder, 0.17),
    (Kind::Other, 0.15),
];

fn draw_kind(rng: &mut Rng) -> Kind {
    let mut x = rng.f64();
    for (k, p) in KIND_SHARES {
        if x < p {
            return k;
        }
        x -= p;
    }
    Kind::Other
}

/// One generated text and its labels.
#[derive(Clone, Debug)]
pub struct Case {
    pub template: &'static str,
    pub state: String,
    pub value: Value,
    /// The value's line (0-based).
    pub line: usize,
    pub n_lines: usize,
}

pub fn case(rng: &mut Rng, templates: &[(&'static str, Template)]) -> Case {
    let kind = draw_kind(rng);
    let value = value(rng, kind);
    let (name, t) = templates[rng.below(templates.len())];
    let (lines, line) = t(rng, &value);
    Case {
        template: name,
        state: lines.join("\n"),
        n_lines: lines.len(),
        value,
        line,
    }
}

pub const Q_SECRET: &str = "Does the text expose a usable secret (an API key, token, password or \
                            private key) that should be revoked?";
pub const Q_SERVICE: &str = "Which service issued the key or token in the text?";
pub const Q_KIND: &str = "What kind of value is assigned or sent in the text?";
pub const Q_LINE: &str = "Which line exposes the secret?";

/// The questions on a case, as records of source `<src>:<question>` (`secret_bench:secret`, …),
/// so `s2-eval`'s per-source table scores each question on its own.
pub fn items(src: &str, c: &Case) -> Vec<Item> {
    let mut recs = Vec::with_capacity(4);
    let secret = c.value.kind == Kind::Secret;
    let s = |q: &str| format!("{src}:{q}");
    recs.push(Question::noul(Q_SECRET).record(&s("secret"), &c.state, one_hot(2, secret as usize)));
    let services: Vec<(String, String)> = SERVICES
        .iter()
        .map(|(k, d)| (k.to_string(), d.to_string()))
        .collect();
    recs.push(Question::choice(Q_SERVICE, services).record(
        &s("service"),
        &c.state,
        one_hot(SERVICES.len(), c.value.service),
    ));
    let kinds: Vec<(String, String)> = KINDS
        .iter()
        .map(|(k, d)| (k.to_string(), d.to_string()))
        .collect();
    recs.push(Question::choice(Q_KIND, kinds).record(
        &s("kind"),
        &c.state,
        one_hot(KINDS.len(), c.value.kind.index()),
    ));
    if c.n_lines >= 2 {
        let mut lines: Vec<(String, String)> = (1..=c.n_lines)
            .map(|i| (format!("line {i}"), String::new()))
            .collect();
        lines.push(("none".into(), "no line exposes a secret".into()));
        let target = if secret { c.line } else { c.n_lines };
        let n = lines.len();
        recs.push(Question::choice(Q_LINE, lines).record(&s("line"), &c.state, one_hot(n, target)));
    }
    let group = state_group(&c.state);
    recs.into_iter()
        .map(|rec| Item {
            group: group.clone(),
            rec,
        })
        .collect()
}

/// Texts per source: each gives three or four records.
pub const SYNTH_TEXTS: usize = 50_000;
pub const BENCH_TEXTS: usize = 2_000;

fn generate(src: &str, seed: u64, n: usize, templates: &[(&'static str, Template)]) -> Vec<Item> {
    let mut rng = Rng::derived(seed, src);
    (0..n)
        .flat_map(|_| items(src, &case(&mut rng, templates)))
        .collect()
}

/// Writes `<out>/secret_synth.jsonl` (train / val by text) and `<out>/secret_bench.jsonl`
/// (test). The jev has its own directory and model (`s6-secrets-pico`): its states run to a
/// few hundred tokens, far past the 64-token state window of the general s6, so they stay out
/// of `data/decisions`.
pub fn write_jev(out: &Path, seed: u64, val_frac: f64) -> Result<Vec<SourceReport>> {
    std::fs::create_dir_all(out)?;
    let opts = DecisionOptions {
        out: out.to_path_buf(),
        seed,
        cap: usize::MAX,
        val_frac,
        only: Vec::new(),
        teacher: PathBuf::new(),
    };
    let sources = [
        ("secret_synth", SYNTH_TEXTS, &TRAIN_TEMPLATES[..], false),
        ("secret_bench", BENCH_TEXTS, &BENCH_TEMPLATES[..], true),
    ];
    sources
        .into_iter()
        .map(|(src, n, templates, test)| {
            let items = generate(src, seed, n, templates);
            finalize(src, items, test, &opts, &out.join(format!("{src}.jsonl")))
        })
        .collect()
}

// ----------------------------------------------------------------------------- baseline

/// Shannon entropy in bits per character.
fn entropy(s: &str) -> f64 {
    let mut counts = [0usize; 256];
    for b in s.bytes() {
        counts[b as usize] += 1;
    }
    let n = s.len() as f64;
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n;
            -p * p.log2()
        })
        .sum()
}

/// The longest run of `set` characters at the start of `s`.
fn run_len(s: &str, set: &[u8]) -> usize {
    s.bytes().take_while(|b| set.contains(b)).count()
}

/// Prefix rules: (prefix, body charset, minimum body length, service).
const RULES: [(&str, &[u8], usize, &str); 20] = [
    ("AKIA", UPPER_DIGITS, 16, "aws"),
    ("ghp_", ALNUM, 36, "github"),
    ("gho_", ALNUM, 36, "github"),
    ("ghu_", ALNUM, 36, "github"),
    ("ghs_", ALNUM, 36, "github"),
    ("github_pat_", URLSAFE, 60, "github"),
    ("glpat-", URLSAFE, 20, "gitlab"),
    ("sk_live_", ALNUM, 24, "stripe"),
    ("rk_live_", ALNUM, 24, "stripe"),
    ("xoxb-", URLSAFE, 30, "slack"),
    ("xoxp-", URLSAFE, 30, "slack"),
    (
        "hooks.slack.com/services/T",
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/",
        40,
        "slack",
    ),
    ("sk-ant-api03-", URLSAFE, 90, "anthropic"),
    ("sk-proj-", URLSAFE, 40, "openai"),
    ("sk-", ALNUM, 40, "openai"),
    ("AIza", URLSAFE, 35, "google"),
    ("hf_", ALPHA, 30, "huggingface"),
    (
        "SG.",
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.",
        60,
        "sendgrid",
    ),
    ("npm_", ALNUM, 36, "npm"),
    ("PRIVATE KEY-----", b"\\n", 2, "private_key"),
];

/// Names a generic rule treats as secret.
const SECRET_WORDS: [&str; 6] = ["key", "secret", "token", "password", "auth", "credential"];

/// Values the generic rule lets through, as gitleaks' allowlists do: documentation keys, and
/// test or publishable keys.
const ALLOWED: [&str; 5] = ["EXAMPLE", "pk_live_", "pk_test_", "sk_test_", "rk_test_"];

/// The `name = value` / `name: value` pairs of a line (`:=`, quotes and `Bearer` included).
fn assignments(line: &str) -> impl Iterator<Item = (&str, &str)> {
    let ident = |c: char| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.');
    line.char_indices()
        .filter(|&(_, c)| c == '=' || c == ':')
        .filter_map(move |(i, _)| {
            let head = line[..i].trim_end().trim_end_matches(['"', '\'']);
            let name = &head[head.rfind(|c: char| !ident(c)).map_or(0, |j| j + 1)..];
            let mut rest = line[i + 1..].trim_start_matches(['=', ' ', '"', '\'']);
            if let Some(r) = rest.strip_prefix("Bearer ") {
                rest = r;
            }
            let end = rest
                .find(|c: char| {
                    c.is_whitespace() || matches!(c, '"' | '\'' | ',' | ';' | ')' | '}')
                })
                .unwrap_or(rest.len());
            (!name.is_empty()).then(|| (name, &rest[..end]))
        })
}

/// The rule scanner: the service of the first live secret it finds in `state`, or `None`.
/// Prefix rules first (with the `EXAMPLE` allowance of AWS), then connection strings with an
/// inline password, then a value of at least 20 characters and 3.5 bits/char of entropy
/// assigned to a secret-sounding name.
pub fn baseline(state: &str) -> Option<&'static str> {
    for (prefix, set, min, svc) in RULES {
        let mut from = 0;
        while let Some(i) = state[from..].find(prefix) {
            let at = from + i + prefix.len();
            let body = &state[at..];
            let n = run_len(body, set);
            // AWS documentation keys end in EXAMPLE
            if n >= min && !(svc == "aws" && body[..n].ends_with("EXAMPLE")) {
                return Some(svc);
            }
            from = at;
        }
    }
    for line in state.lines() {
        if let Some(i) = line.find("://") {
            let rest = &line[i + 3..];
            if let (Some(c), Some(a)) = (rest.find(':'), rest.find('@')) {
                let pass = &rest[c + 1..a.max(c + 1)];
                if c < a && !pass.is_empty() && !pass.starts_with(['$', '{', '<', '*']) {
                    return Some("database");
                }
            }
        }
        for (name, v) in assignments(line) {
            let name = name.to_ascii_lowercase();
            if SECRET_WORDS.iter().any(|w| name.contains(w))
                && v.len() >= 20
                && entropy(v) >= 3.5
                // a digit: an identifier (`api_key=stripe_key`) is a reference, not a value
                && v.bytes().any(|b| b.is_ascii_digit())
                && v.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"+/=-_.".contains(&b))
                && !ALLOWED.iter().any(|a| v.contains(a))
            {
                return Some("generic");
            }
        }
    }
    None
}

/// The rule scanner's scores on decision records of this source (`secret_bench.jsonl`): the
/// secret question as a detector, and the service question.
#[derive(Debug, Default, PartialEq)]
pub struct BaselineReport {
    pub texts: usize,
    pub tp: usize,
    pub fp: usize,
    pub fn_: usize,
    pub tn: usize,
    pub service_ok: usize,
    /// False positives by kind ([`KINDS`] order) and misses by service ([`SERVICES`] order).
    pub fp_by_kind: [usize; KINDS.len()],
    pub missed_by_service: [usize; SERVICES.len()],
}

pub fn baseline_report(records: &[crate::record::DecisionRecord]) -> BaselineReport {
    use std::collections::BTreeMap;
    let argmax = |r: &crate::record::DecisionRecord| crate::record::argmax(&r.target);
    // state -> (secret, service, kind)
    let mut by_state: BTreeMap<&str, [Option<usize>; 3]> = BTreeMap::new();
    for r in records {
        let slot = match r.ins.as_str() {
            Q_SECRET => 0,
            Q_SERVICE => 1,
            Q_KIND => 2,
            _ => continue,
        };
        by_state.entry(&r.state).or_default()[slot] = Some(argmax(r));
    }
    let mut rep = BaselineReport::default();
    for (state, [secret, svc, kind]) in by_state {
        let (Some(secret), Some(svc)) = (secret, svc) else {
            continue;
        };
        rep.texts += 1;
        let found = baseline(state);
        match (secret == 1, found.is_some()) {
            (true, true) => rep.tp += 1,
            (true, false) => {
                rep.fn_ += 1;
                rep.missed_by_service[svc] += 1;
            }
            (false, true) => {
                rep.fp += 1;
                if let Some(k) = kind {
                    rep.fp_by_kind[k] += 1;
                }
            }
            (false, false) => rep.tn += 1,
        }
        // a scanner names a service only for what it reports
        if found.map_or(NONE, service) == svc {
            rep.service_ok += 1;
        }
    }
    rep
}

impl std::fmt::Display for BaselineReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n = self.texts.max(1) as f64;
        let ratio = |a: usize, b: usize| a as f64 / (b.max(1)) as f64;
        writeln!(f, "rule scanner on {} texts", self.texts)?;
        writeln!(
            f,
            "  secret: accuracy {:.3}, precision {:.3}, recall {:.3} (tp {} fp {} fn {} tn {})",
            (self.tp + self.tn) as f64 / n,
            ratio(self.tp, self.tp + self.fp),
            ratio(self.tp, self.tp + self.fn_),
            self.tp,
            self.fp,
            self.fn_,
            self.tn
        )?;
        writeln!(f, "  service: accuracy {:.3}", self.service_ok as f64 / n)?;
        let fps: Vec<String> = KINDS
            .iter()
            .zip(self.fp_by_kind)
            .filter(|(_, c)| *c > 0)
            .map(|((k, _), c)| format!("{k} {c}"))
            .collect();
        writeln!(f, "  false positives by kind: {}", fps.join(", "))?;
        let missed: Vec<String> = SERVICES
            .iter()
            .zip(self.missed_by_service)
            .filter(|(_, c)| *c > 0)
            .map(|((k, _), c)| format!("{k} {c}"))
            .collect();
        write!(f, "  missed secrets by service: {}", missed.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::validate_record;

    fn cases(n: usize, templates: &[(&'static str, Template)]) -> Vec<Case> {
        let mut rng = Rng::derived(7, "test");
        (0..n).map(|_| case(&mut rng, templates)).collect()
    }

    #[test]
    fn records_are_valid_and_deterministic() {
        let a = generate("secret_synth", 3, 300, &TRAIN_TEMPLATES);
        let b = generate("secret_synth", 3, 300, &TRAIN_TEMPLATES);
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(&b).all(|(x, y)| x.rec == y.rec));
        for it in &a {
            validate_record(&it.rec).unwrap();
            // one source per question, for the per-source table of s2-eval
            let q = it.rec.src.strip_prefix("secret_synth:").unwrap();
            assert!(["secret", "service", "kind", "line"].contains(&q), "{q}");
        }
        // three or four questions per text, all on the same state group
        assert!(a.len() >= 900 && a.len() <= 1200, "{}", a.len());
    }

    #[test]
    fn the_value_is_on_its_line_and_states_stay_short() {
        for c in cases(2000, &TRAIN_TEMPLATES)
            .into_iter()
            .chain(cases(500, &BENCH_TEMPLATES))
        {
            let lines: Vec<&str> = c.state.lines().collect();
            assert_eq!(lines.len(), c.n_lines);
            assert!(
                lines[c.line].contains(&c.value.text),
                "{}: {:?}",
                c.template,
                c.state
            );
            assert_eq!(c.state.matches(&c.value.text).count(), 1, "{}", c.state);
            // the model reads ~1000 characters at most
            assert!(c.state.len() <= 700, "{} chars: {}", c.state.len(), c.state);
        }
    }

    #[test]
    fn labels_are_consistent_and_balanced() {
        let cs = cases(4000, &TRAIN_TEMPLATES);
        let secret = cs.iter().filter(|c| c.value.kind == Kind::Secret).count();
        assert!((1800..2200).contains(&secret), "{secret}");
        for c in &cs {
            // only live, test and public keys name a service
            let named = c.value.service != NONE;
            assert_eq!(
                named,
                matches!(c.value.kind, Kind::Secret | Kind::Test | Kind::Public),
                "{c:?}"
            );
        }
        // every service issues secrets
        for (s, (name, _)) in SERVICES.iter().enumerate().take(LIVE_SERVICES) {
            assert!(
                cs.iter()
                    .any(|c| c.value.kind == Kind::Secret && c.value.service == s),
                "{name}"
            );
        }
    }

    #[test]
    fn bench_contexts_are_never_trained_on() {
        let train: Vec<&str> = TRAIN_TEMPLATES.iter().map(|t| t.0).collect();
        assert!(BENCH_TEMPLATES.iter().all(|t| !train.contains(&t.0)));
        let synth = generate("secret_synth", 1, 500, &TRAIN_TEMPLATES);
        let bench = generate("secret_bench", 1, 200, &BENCH_TEMPLATES);
        let groups: std::collections::HashSet<_> = synth.iter().map(|i| &i.group).collect();
        assert!(bench.iter().all(|i| !groups.contains(&i.group)));
    }

    #[test]
    fn baseline_report_counts_every_text() {
        let recs: Vec<_> = generate("secret_bench", 5, 400, &BENCH_TEMPLATES)
            .into_iter()
            .map(|i| i.rec)
            .collect();
        let rep = baseline_report(&recs);
        assert_eq!(rep.tp + rep.fp + rep.fn_ + rep.tn, rep.texts);
        assert!(rep.texts > 350, "{}", rep.texts);
        // the prefixed keys are found; the rules are not the whole task
        assert!(rep.tp > rep.fn_, "{rep}");
        assert!(rep.fp + rep.fn_ > 0, "{rep}");
    }

    #[test]
    fn the_baseline_finds_prefixed_keys_and_skips_hard_negatives() {
        let mut rng = Rng::derived(11, "baseline");
        for svc in ["aws", "github", "stripe", "anthropic", "huggingface", "npm"] {
            let (key, _) = live(&mut rng, svc);
            let v = Value {
                text: key,
                var: "X".into(),
                kind: Kind::Secret,
                service: service(svc),
            };
            let (lines, _) = t_env(&mut rng, &v);
            let found = baseline(&lines.join("\n"));
            // an AWS secret access key has no prefix: the generic rule does not see it under
            // the name `X`
            if svc != "aws" || v.text.starts_with("AKIA") {
                assert_eq!(found, Some(svc), "{}", v.text);
            }
        }
        for text in [
            "STRIPE_PUBLISHABLE_KEY=pk_live_0123456789abcdefABCDEFGH",
            "AWS_ACCESS_KEY_ID=AKIA012345678EXAMPLE",
            "OPENAI_API_KEY=${OPENAI_API_KEY}",
            "GITHUB_TOKEN=ghp_************************",
            "BUILD_SHA=3f786850e387550fdab836ed7e6dc881de23001b",
        ] {
            assert_eq!(baseline(text), None, "{text}");
        }
        assert_eq!(
            baseline("DATABASE_URL=postgres://app:s3cr3tPassw0rd@db:5432/main"),
            Some("database")
        );
        assert_eq!(
            baseline("DATABASE_URL=postgres://app:${DB_PASS}@db/main"),
            None
        );
    }
}
