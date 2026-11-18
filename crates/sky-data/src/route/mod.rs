//! The data of s6-route-nano, a tool router that reads tool descriptions
//! (`docs/s6-nano2-prereg.md`, section 4): `sky-data route <step>`.
//!
//! - [`sources`]: every source read into one table of tools and one of requests (4.1, 4.3);
//! - [`cluster`]: tool names, clusters and the train / validation / test split by cluster (4.2);
//! - [`slates`](mod@slates): the option lists (4.4): the four training kinds, the test-like slates and the
//!   files of the reading checks (7.3);
//! - [`build`]: the steps in order and `BUILD.json`;
//! - [`fresh`]: the fresh test files of v3 (`docs/s6-nano3-prereg.md`);
//! - [`v4`]: the data of s6-route v4 (`docs/s6-route4-prereg.md`, section 4), from v2's build;
//! - [`toucan`]: the fresh test of s6-route v4 (section 6), from Toucan-1.5M, and of v5
//!   (`docs/s6-route5-prereg.md` 6, `--rule v5`);
//! - [`v5`]: the data of s6-route v5 (`docs/s6-route5-prereg.md`, section 4), from v2's build
//!   and v4's rewrites.
//!
//! Every file goes to the data folder, outside the repository. The test files are written once
//! and never read again by this module; the steps print counts only.

pub mod build;
pub mod cluster;
pub mod fresh;
pub mod slates;
pub mod sources;
pub mod toucan;
pub mod v4;
pub mod v5;

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

pub use crate::secrets2::{read_json, sha256_file, sha256_hex, write_json};

/// The instruction of every record, word for word (section 3).
pub const INSTRUCTION: &str = "Which tool should handle the user's request?";
/// The last option of every slate (section 3).
pub const NONE_OPTION: &str = "none of these tools fits the request";
/// The salt of the cluster split (4.2, item 3) and of the MASSIVE hold-out (4.3).
pub const SALT: &str = "1234";
/// A description is cut to its first sentence and at most this many words (section 3).
pub const DESC_WORDS: usize = 30;

/// The three training groups (4.1): one record directory each, the trainer's `--weights`.
pub const TOOLS: &str = "tools";
pub const WHEN2CALL: &str = "when2call";
pub const INTENTS: &str = "intents";
/// The group of the requests of the sources that are only tested.
pub const TEST: &str = "test";

/// One tool as the router sees it: a name and the first sentence of its description.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Tool {
    /// `t` and 16 hex digits of the sha256 of the option text.
    pub id: String,
    pub name: String,
    /// The description, cut (section 3) and with key-like strings masked (4.6, item 5).
    pub desc: String,
}

impl Tool {
    pub fn new(name: &str, raw_desc: &str) -> Self {
        let name = collapse_ws(name);
        let desc = cut_description(&mask_keys(raw_desc).0);
        let id = tool_id(&option_text(&name, &desc));
        Self { id, name, desc }
    }

    /// `name: description` (the name alone without a description).
    pub fn option(&self) -> String {
        option_text(&self.name, &self.desc)
    }

    /// The normalised name of 4.2, item 1.
    pub fn norm(&self) -> String {
        norm_name(&self.name)
    }
}

/// One request with what its source says of it.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Request {
    /// `<source>:<row>`.
    pub id: String,
    pub source: String,
    /// [`TOOLS`], [`WHEN2CALL`], [`INTENTS`] or [`TEST`].
    pub group: String,
    /// The split the source fixes (When2Call, the intents, the test-only sources); `None` for
    /// `route/tools`, whose split is the one of its gold tool's cluster.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split: Option<String>,
    pub lang: String,
    /// The user's request (one turn), with key-like strings masked.
    pub user: String,
    /// The tools the source lists with the request, in its order (tool ids).
    #[serde(default)]
    pub tools: Vec<String>,
    /// The tool that should handle it; `None`: none of the tools fits.
    pub gold: Option<String>,
    /// Other tools the source calls for the same request: never a distractor.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exclude: Vec<String>,
    /// `call`, `request_for_info`, `cannot_answer`, `direct_answer`, `intent` or `out_of_scope`.
    pub kind: String,
}

/// Whitespace runs collapsed to one space, trimmed.
pub fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Section 3: the first sentence of a description (up to the first `.`, `!` or `?` followed by
/// a space or the end), then at most [`DESC_WORDS`] words.
pub fn cut_description(d: &str) -> String {
    let d = collapse_ws(d);
    let chars: Vec<char> = d.chars().collect();
    let mut end = chars.len();
    for (i, c) in chars.iter().enumerate() {
        if matches!(c, '.' | '!' | '?') && chars.get(i + 1).is_none_or(|n| *n == ' ') {
            end = i + 1;
            break;
        }
    }
    let first: String = chars[..end].iter().collect();
    first
        .split(' ')
        .filter(|w| !w.is_empty())
        .take(DESC_WORDS)
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn option_text(name: &str, desc: &str) -> String {
    if desc.is_empty() {
        name.to_string()
    } else {
        format!("{name}: {desc}")
    }
}

/// `t` and the first 16 hex digits of the sha256 of the option text.
pub fn tool_id(option: &str) -> String {
    format!("t{}", &sha256_hex(option.as_bytes())[..16])
}

/// The leading words dropped from a tool name (4.2, item 1).
pub const NAME_VERBS: [&str; 7] = [
    "get", "fetch", "retrieve", "find", "search", "lookup", "query",
];

/// 4.2, item 1: lower-case; the words are the runs of letters and digits (every other
/// character is a separator); a leading word of [`NAME_VERBS`] is dropped when another word
/// follows; the words joined by `_`.
pub fn norm_name(name: &str) -> String {
    let lower = name.to_lowercase();
    let mut words: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    if words.len() > 1 && NAME_VERBS.contains(&words[0]) {
        words.remove(0);
    }
    words.join("_")
}

/// The normalised text of a request (4.2, item 4; 4.3; 4.6): lower-case, every run of
/// characters that are not letters or digits made one space, trimmed.
pub fn norm_text(s: &str) -> String {
    let lower = s.to_lowercase();
    lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// The state of a record (section 3): the request prefixed `User: `.
pub fn state_of(user: &str) -> String {
    format!("User: {}", user.trim())
}

/// Prefixes of provider keys (4.6, item 5).
const KEY_PREFIXES: [&str; 14] = [
    "sk-",
    "sk_live_",
    "sk_test_",
    "pk_live_",
    "rk_live_",
    "AKIA",
    "ASIA",
    "ghp_",
    "gho_",
    "github_pat_",
    "xoxb-",
    "xoxp-",
    "AIza",
    "glpat-",
];

/// The mask written in place of a key-like string.
pub const KEY_MASK: &str = "<KEY>";

fn entropy_bits(s: &str) -> f64 {
    let mut counts: BTreeMap<char, usize> = BTreeMap::new();
    for c in s.chars() {
        *counts.entry(c).or_default() += 1;
    }
    let n = s.chars().count() as f64;
    counts
        .values()
        .map(|&k| {
            let p = k as f64 / n;
            -p * p.log2()
        })
        .sum()
}

/// The characters of a token that [`mask_keys`] looks at.
fn key_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || "-_+/".contains(c)
}

/// Whether a token looks like a live key: a provider prefix followed by at least 12 more
/// characters, or at least 24 characters of letters, digits and `-_+/` with digits, both cases
/// of letters, and at least 4 bits of entropy per character.
pub fn looks_like_key(t: &str) -> bool {
    let body_ok = |s: &str| s.chars().all(key_char);
    if KEY_PREFIXES
        .iter()
        .any(|p| t.starts_with(p) && t.len() >= p.len() + 12 && body_ok(t))
    {
        return true;
    }
    t.len() >= 24
        && body_ok(t)
        && t.chars().any(|c| c.is_ascii_digit())
        && t.chars().any(|c| c.is_ascii_lowercase())
        && t.chars().any(|c| c.is_ascii_uppercase())
        && entropy_bits(t) >= 4.0
}

/// `s` with every key-like token ([`looks_like_key`]; tokens are the runs of letters, digits
/// and `-_+/`) replaced by [`KEY_MASK`], and the number replaced.
pub fn mask_keys(s: &str) -> (String, usize) {
    let is_tok = key_char;
    let mut out = String::with_capacity(s.len());
    let mut n = 0;
    let mut tok = String::new();
    let mut flush = |tok: &mut String, out: &mut String| {
        if !tok.is_empty() {
            if looks_like_key(tok) {
                out.push_str(KEY_MASK);
                n += 1;
            } else {
                out.push_str(tok);
            }
            tok.clear();
        }
    };
    for c in s.chars() {
        if is_tok(c) {
            tok.push(c);
        } else {
            flush(&mut tok, &mut out);
            out.push(c);
        }
    }
    flush(&mut tok, &mut out);
    (out, n)
}

/// A counter with sorted keys, for the counts written to JSON.
pub type Counts = BTreeMap<String, usize>;

pub fn bump(c: &mut Counts, key: impl AsRef<str>) {
    *c.entry(key.as_ref().to_string()).or_default() += 1;
}

pub fn add(c: &mut Counts, key: impl AsRef<str>, n: usize) {
    *c.entry(key.as_ref().to_string()).or_default() += n;
}

/// Reads a JSONL file of `T`.
pub fn read_lines<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    crate::record::read_jsonl(path).with_context(|| format!("read {}", path.display()))
}

/// sha256 of `<key><salt>`, as the split and the hold-out read it.
pub fn salted(key: &str) -> [u8; 32] {
    let d = ring::digest::digest(&ring::digest::SHA256, format!("{key}{SALT}").as_bytes());
    let mut out = [0u8; 32];
    out.copy_from_slice(d.as_ref());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptions_are_cut_to_a_sentence_and_thirty_words() {
        assert_eq!(
            cut_description("  Get the  weather.  It is free. "),
            "Get the weather."
        );
        assert_eq!(
            cut_description("Version 2.5 of the API"),
            "Version 2.5 of the API"
        );
        assert_eq!(cut_description("What? No"), "What?");
        let long = (0..40)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(cut_description(&long).split(' ').count(), DESC_WORDS);
        assert_eq!(cut_description(""), "");
    }

    #[test]
    fn names_are_normalised() {
        assert_eq!(norm_name("get_current_weather"), "current_weather");
        assert_eq!(norm_name("Get Stock Price"), "stock_price");
        assert_eq!(norm_name("fetch--news__headlines"), "news_headlines");
        assert_eq!(norm_name("search"), "search");
        assert_eq!(norm_name("getWeather"), "getweather");
        assert_eq!(norm_name("/madlibs-diceware"), "madlibs_diceware");
        // one leading verb only
        assert_eq!(norm_name("get_search_results"), "search_results");
    }

    #[test]
    fn texts_are_normalised() {
        assert_eq!(
            norm_text("  What's the WEATHER, today?? "),
            "what s the weather today"
        );
        assert_eq!(norm_text("Réveille-moi à 7h"), "réveille moi à 7h");
    }

    #[test]
    fn tools_carry_their_option_text() {
        let t = Tool::new(" get_news ", "Get the latest news. Updated hourly.");
        assert_eq!(t.option(), "get_news: Get the latest news.");
        assert_eq!(t.id, tool_id("get_news: Get the latest news."));
        assert_eq!(t.norm(), "news");
        assert_eq!(Tool::new("ping", "").option(), "ping");
    }

    #[test]
    fn keys_are_masked() {
        let (s, n) = mask_keys("use key sk-AbCdEf0123456789xyz now");
        assert_eq!((s.as_str(), n), ("use key <KEY> now", 1));
        let (s, n) = mask_keys("token=Zq8Lm2Nx7Rt4Vb9Kc3Wd6Hf1Jg5Ps0Ty");
        assert_eq!((s.as_str(), n), ("token=<KEY>", 1));
        // identifiers, words and plain numbers stay
        for keep in [
            "get_current_weather_by_city_name",
            "2026-11-15T10:00:00",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "the quick brown fox",
        ] {
            assert_eq!(mask_keys(keep), (keep.to_string(), 0), "{keep}");
        }
    }

    #[test]
    fn states_are_prefixed() {
        assert_eq!(
            state_of(" turn on the lights \n"),
            "User: turn on the lights"
        );
    }
}
