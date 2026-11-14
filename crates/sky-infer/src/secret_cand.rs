//! s6-secrets v2 (`docs/s6-secrets2-prereg.md`, section 4): the state of one rule candidate.
//!
//! A rule engine proposes a candidate (a value, its line, its neighbours, its file); the model
//! judges it. This module turns the candidate record into the state text the model reads, with
//! the hand features of section 4.2 as `name value` pairs (one fused number token per value,
//! SPEC-S3 section 1), and holds the two questions of section 1. It is pure: no file is read,
//! the tokenizer and the dictionary are given. The data tool (`sky-data secrets2`) and the engine
//! call the same functions, and `tests/fixtures/secret_cand.json` holds inputs and expected
//! outputs (path scopes, features, full states) that any other implementation must reproduce.
//!
//! Units. A candidate record counts columns in UTF-16 code units, as its generator does
//! (JavaScript string indices). Every length and every cut of this module counts Unicode code
//! points, so a cut never splits a character; the entropy alone follows the generator's
//! `shannonEntropy` to the letter (see [`entropy`]).

use crate::tokenizer::Tokenizer;

/// The gated question (noul): T rows are true, F and X rows are false.
pub const Q_REAL: &str = "Is this value a real credential that should be revoked?";

/// The reported question (choice over [`KINDS`]). Section 1 gives its three options word for
/// word and no sentence for the question itself; this is the sentence.
pub const Q_KIND: &str = "What kind of value is this?";

/// The options of [`Q_KIND`], word for word (section 1): CredData `T`, `X` and `F`.
pub const KINDS: [&str; 3] = [
    "a real credential",
    "an example, test value or placeholder",
    "not a credential",
];

/// CredData's `word_in_path.json` at commit `01e3d135a90f46c9a7965683b24a05d12e136b33`
/// (Apache-2.0), in its order: the words [`file_scope`] looks for in a path.
pub const WORD_IN_PATH: [&str; 57] = [
    "test",
    "mock",
    "/src",
    "code",
    "/include",
    "internal",
    "tool",
    "util",
    "example",
    "sample",
    "conf",
    "secret",
    "setting",
    "security",
    "secure",
    "resource",
    "fixture",
    "docker",
    "/docs",
    "/doc/",
    "document",
    "/lang",
    "/local/",
    "/locale",
    "/lib",
    "/spec",
    "/pkg",
    "/api",
    "/rest",
    "/opt",
    "/sys",
    "kube",
    "kafka",
    "cluster",
    "template",
    "other",
    "public",
    "init",
    "client",
    "server",
    "/model",
    "/modul",
    "browser",
    "/env/",
    "/app",
    "/assets/",
    "vendor",
    "build",
    "/record",
    "/script",
    "python",
    "/usr",
    "/etc",
    "/other",
    "/fuzz",
    "/dist-packages",
    "/site-packages",
];

/// Scope words written in a state (amendment 1 of the pre-registration).
pub const MAX_SCOPE_WORDS: usize = 4;

/// The scope written when a path holds no word of the list.
pub const NO_SCOPE: &str = "_";

/// Port of CredData's `get_file_scope()` (`download_data.py`): the directory chain CredData files
/// a path under, e.g. `/test/src/conf/` for `src/test/resources/dockerConfig/…`-like paths, and
/// `/_/` when the path holds no word of [`WORD_IN_PATH`]. `path_without_extension` is the path
/// relative to its repository, with `/` separators and no extension ([`split_extension`]).
pub fn file_scope(path_without_extension: &str) -> String {
    let local = format!("./{}", path_without_extension.to_lowercase());
    let mut result = String::from("/");
    for word in WORD_IN_PATH {
        if local.contains(word) {
            result.push_str(word.strip_prefix('/').unwrap_or(word));
            if !result.ends_with('/') {
                result.push('/');
            }
        }
    }
    if result == "/" {
        result.push_str("_/");
    }
    result
}

/// The words of [`file_scope`], in its order; empty when there is none (`/_/`).
pub fn scope_words(path_without_extension: &str) -> Vec<&'static str> {
    let local = format!("./{}", path_without_extension.to_lowercase());
    WORD_IN_PATH
        .iter()
        .filter(|w| local.contains(**w))
        .map(|w| w.trim_matches('/'))
        .collect()
}

/// Python's `os.path.splitext` on a `/`-separated path, the extension lower-cased as CredData
/// does: the last dot of the last component starts the extension, unless that component only
/// has leading dots (`.env` has no extension).
pub fn split_extension(path: &str) -> (&str, String) {
    let base = path.rfind('/').map_or(0, |i| i + 1);
    let name = &path[base..];
    let lead = name.len() - name.trim_start_matches('.').len();
    match name[lead..].rfind('.') {
        Some(i) => {
            let at = base + lead + i;
            (&path[..at], path[at..].to_lowercase())
        }
        None => (path, String::new()),
    }
}

/// Scope words and extension of a file from its real path, as a deployed host computes them:
/// [`split_extension`], then [`scope_words`] of the path without its extension. The path is
/// relative to its repository, as CredData's are: an absolute path would bring the words of
/// the directories above the repository (`/usr`, `/local/`…) into the scope.
pub fn path_scope(path: &str) -> (Vec<&'static str>, String) {
    let (stem, ext) = split_extension(path);
    (scope_words(stem), ext)
}

/// The columns of the value in `lineText` from the fields of a candidate record (`line`,
/// `endLine`, `start`, `end`, `lineFrom`), for [`Candidate::value_start`] and
/// [`Candidate::value_end`]: a value that ends on a later line runs to the end of its first.
pub fn value_columns(
    line: u64,
    end_line: u64,
    start: usize,
    end: usize,
    line_from: usize,
) -> (usize, usize) {
    let a = start.saturating_sub(line_from);
    if end_line > line {
        (a, usize::MAX)
    } else {
        (a, end.saturating_sub(line_from).max(a))
    }
}

/// The filters of the default engine that the candidate generator turns off, as 0/1 features:
/// whether the default engine would have dropped the value for each (pre-registration 2.1, 4.2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Filters {
    pub entropy: bool,
    pub placeholder: bool,
    pub stopword: bool,
    pub allowlist: bool,
}

/// One rule candidate, in the terms of the generator's record (`sg-cand-1`).
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate<'a> {
    /// Rule id.
    pub rule: &'a str,
    /// Scope words of the file's path, in [`file_scope`]'s order; at most [`MAX_SCOPE_WORDS`]
    /// are written. Empty: [`NO_SCOPE`].
    pub scope: &'a [&'a str],
    /// The file's extension, lower-case, with its dot; empty when none.
    pub ext: &'a str,
    /// The variable or key name.
    pub name: Option<&'a str>,
    pub value: &'a str,
    /// The line of the value's first character, without its line ending (or a window of it).
    pub line_text: &'a str,
    /// Column of the value in `line_text`, in UTF-16 code units (`start - lineFrom`).
    pub value_start: usize,
    /// Column after the value in `line_text`, in UTF-16 code units; past the line's end when
    /// the value runs over several lines.
    pub value_end: usize,
    pub prev_line: Option<&'a str>,
    pub next_line: Option<&'a str>,
    pub filters: Filters,
}

/// The dictionary of the pre-registration (section 3.2): the lower-case words of 4 to 12
/// letters of a word list.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WordSet {
    /// Sorted by length, then by spelling.
    words: Vec<Box<str>>,
}

impl WordSet {
    pub const MIN_LEN: usize = 4;
    pub const MAX_LEN: usize = 12;

    /// The entries of a word list (one per line) that are written with the letters `a` to `z`
    /// only and have 4 to 12 of them. An entry with a capital (a proper noun in
    /// `/usr/share/dict/words`) is left out, not lower-cased.
    pub fn from_text(text: &str) -> Self {
        let mut words: Vec<Box<str>> = text
            .lines()
            .map(|l| l.trim_end_matches('\r'))
            .filter(|w| {
                (Self::MIN_LEN..=Self::MAX_LEN).contains(&w.len())
                    && w.bytes().all(|b| b.is_ascii_lowercase())
            })
            .map(Box::from)
            .collect();
        words.sort_unstable_by(|a, b| (a.len(), a).cmp(&(b.len(), b)));
        words.dedup();
        Self { words }
    }

    pub fn len(&self) -> usize {
        self.words.len()
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// The words, by length then spelling.
    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.words.iter().map(|w| &**w)
    }

    /// Whether `word` (lower-case) is in the dictionary.
    pub fn contains(&self, word: &str) -> bool {
        self.words
            .binary_search_by(|w| (w.len(), &**w).cmp(&(word.len(), word)))
            .is_ok()
    }

    /// The words of `n` letters, sorted.
    pub fn of_len(&self, n: usize) -> &[Box<str>] {
        let from = self.words.partition_point(|w| w.len() < n);
        let to = self.words.partition_point(|w| w.len() <= n);
        &self.words[from..to]
    }

    /// Whether a run of ASCII letters is a word of the dictionary, whatever its case.
    pub fn is_word(&self, run: &str) -> bool {
        if !(Self::MIN_LEN..=Self::MAX_LEN).contains(&run.len()) {
            return false;
        }
        if run.bytes().all(|b| b.is_ascii_lowercase()) {
            return self.contains(run);
        }
        self.contains(&run.to_ascii_lowercase())
    }
}

/// The maximal runs of ASCII letters of `s`, as byte ranges.
pub fn letter_runs(s: &str) -> impl Iterator<Item = (usize, usize)> + '_ {
    let b = s.as_bytes();
    let mut i = 0;
    std::iter::from_fn(move || {
        while i < b.len() && !b[i].is_ascii_alphabetic() {
            i += 1;
        }
        if i == b.len() {
            return None;
        }
        let from = i;
        while i < b.len() && b[i].is_ascii_alphabetic() {
            i += 1;
        }
        Some((from, i))
    })
}

/// Length of the longest dictionary-word run of a value: the longest maximal run of ASCII
/// letters that is, as a whole and whatever its case, a word of the dictionary; 0 when none.
/// It is the run the word-preserving replacement of section 3.2 keeps as a word.
pub fn longest_word_run(value: &str, words: &WordSet) -> usize {
    letter_runs(value)
        .filter(|&(a, b)| words.is_word(&value[a..b]))
        .map(|(a, b)| b - a)
        .max()
        .unwrap_or(0)
}

/// Shannon entropy of a value in bits per character, as secretgate's `src/engine/entropy.ts`
/// computes it: the count of each distinct code point, divided by the length in UTF-16 code
/// units, summed in order of first occurrence (`h -= p * log2(p)`). 0 for an empty value.
pub fn entropy(value: &str) -> f64 {
    let len = value.encode_utf16().count();
    if len == 0 {
        return 0.0;
    }
    // Distinct code points in order of first occurrence. Values are short: a linear search
    // over the distinct ones is enough.
    let mut counts: Vec<(char, u32)> = Vec::new();
    for c in value.chars() {
        match counts.iter_mut().find(|(d, _)| *d == c) {
            Some((_, n)) => *n += 1,
            None => counts.push((c, 1)),
        }
    }
    let mut h = 0.0f64;
    for (_, n) in counts {
        let p = n as f64 / len as f64;
        h -= p * p.log2();
    }
    h
}

/// A non-negative quantity in hundredths, rounded half up: `floor(x * 100 + 0.5)`.
fn hundredths(x: f64) -> u32 {
    (x * 100.0 + 0.5).floor().max(0.0) as u32
}

/// `1234` → `12.34`.
fn write_hundredths(out: &mut String, h: u32) {
    use std::fmt::Write;
    let _ = write!(out, "{}.{:02}", h / 100, h % 100);
}

/// The hand features of one value (pre-registration 4.2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Features {
    /// Length of the value, in code points.
    pub len: u32,
    /// [`entropy`], in hundredths of a bit, rounded half up.
    pub entropy: u32,
    /// Tokens of the value under the model's BPE (the value alone, no number extraction).
    pub tokens: u32,
    /// Code points per token, in hundredths, rounded half up; 0 for an empty value.
    pub chars_per_token: u32,
    /// ASCII digits, ASCII upper-case letters, ASCII lower-case letters, and every other code
    /// point.
    pub digits: u32,
    pub upper: u32,
    pub lower: u32,
    pub other: u32,
    /// [`longest_word_run`].
    pub word_run: u32,
    pub filters: Filters,
}

impl Features {
    /// The features of `value`; `tok` is the model's tokenizer.
    pub fn of(value: &str, tok: &Tokenizer, words: &WordSet, filters: Filters) -> Self {
        let mut f = Self {
            filters,
            ..Self::default()
        };
        for c in value.chars() {
            f.len += 1;
            match c {
                '0'..='9' => f.digits += 1,
                'A'..='Z' => f.upper += 1,
                'a'..='z' => f.lower += 1,
                _ => f.other += 1,
            }
        }
        f.entropy = hundredths(entropy(value));
        f.tokens = tok.encode(value).len() as u32;
        if f.tokens > 0 {
            // round half up, in integers: (100 * len / tokens) + 1/2
            f.chars_per_token =
                ((200 * f.len as u64 + f.tokens as u64) / (2 * f.tokens as u64)) as u32;
        }
        f.word_run = longest_word_run(value, words) as u32;
        f
    }

    /// The features as `name value` pairs, in the order of section 4.2. Each value is one
    /// number for the numeric extraction, so with `fuse_numbers` it costs no token.
    pub fn write(&self, out: &mut String) {
        use std::fmt::Write;
        let _ = write!(out, "len {} ent ", self.len);
        write_hundredths(out, self.entropy);
        out.push_str(" cpt ");
        write_hundredths(out, self.chars_per_token);
        let b = |x: bool| x as u8;
        let _ = write!(
            out,
            " dig {} up {} low {} oth {} word {} gate {} ph {} stop {} allow {}",
            self.digits,
            self.upper,
            self.lower,
            self.other,
            self.word_run,
            b(self.filters.entropy),
            b(self.filters.placeholder),
            b(self.filters.stopword),
            b(self.filters.allowlist),
        );
    }
}

/// Stands for the value in its line.
pub const VALUE_MARK: &str = "<V>";
/// Stands for the middle of a value longer than [`VALUE_MAX`].
pub const ELISION_MARK: &str = "<~>";
/// Characters kept of the name.
pub const NAME_MAX: usize = 40;
/// A value up to this length is written whole; a longer one as its first and last
/// [`VALUE_SIDE`] characters around [`ELISION_MARK`].
pub const VALUE_MAX: usize = 64;
pub const VALUE_SIDE: usize = 32;
/// Characters of the line window, [`VALUE_MARK`] included.
pub const LINE_MAX: usize = 120;
/// Characters kept of the previous and of the next line.
pub const CONTEXT_MAX: usize = 100;

/// The first `n` code points of `s`.
fn head(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// The last `n` code points of `s`.
fn tail(s: &str, n: usize) -> &str {
    if n == 0 {
        return "";
    }
    match s.char_indices().rev().nth(n - 1) {
        Some((i, _)) => &s[i..],
        None => s,
    }
}

/// The byte offset of the UTF-16 column `col` of `s`: the end of `s` when `col` is past it, and
/// the start of the character when `col` falls inside a surrogate pair.
fn utf16_to_byte(s: &str, col: usize) -> usize {
    let mut units = 0;
    for (i, c) in s.char_indices() {
        if units + c.len_utf16() > col {
            return i;
        }
        units += c.len_utf16();
    }
    s.len()
}

/// The line with the value replaced by [`VALUE_MARK`], cut to [`LINE_MAX`] characters centred on
/// the mark: half of the room on each side, and what one side does not use goes to the other.
pub fn line_window(line_text: &str, value_start: usize, value_end: usize) -> String {
    let a = utf16_to_byte(line_text, value_start);
    let b = utf16_to_byte(line_text, value_end.max(value_start)).max(a);
    let (left, right) = (&line_text[..a], &line_text[b..]);
    let room = LINE_MAX - VALUE_MARK.chars().count();
    let (nl, nr) = (left.chars().count(), right.chars().count());
    let mut l = nl.min(room / 2);
    let r = nr.min(room - l);
    l = nl.min(room - r);
    let mut out = String::with_capacity(LINE_MAX * 2);
    out.push_str(tail(left, l));
    out.push_str(VALUE_MARK);
    out.push_str(head(right, r));
    out
}

/// One part of the state: its label, then its text when there is one.
fn part(out: &mut String, label: &str, text: &str) {
    if !out.is_empty() {
        out.push('\n');
    }
    out.push_str(label);
    if !text.is_empty() {
        out.push(' ');
        out.push_str(text);
    }
}

/// The state of a candidate with its features already computed. The parts come in the order of
/// section 4.1, one per line, because a state is truncated from its end: rule, path (scope
/// words and extension), key, value, features, line, previous line, next line.
pub fn state_with(c: &Candidate<'_>, f: &Features) -> String {
    let mut out = String::with_capacity(512);
    part(&mut out, "rule", c.rule);
    let mut path = String::new();
    for w in c.scope.iter().take(MAX_SCOPE_WORDS) {
        if !path.is_empty() {
            path.push(' ');
        }
        path.push_str(w);
    }
    if path.is_empty() {
        path.push_str(NO_SCOPE);
    }
    if !c.ext.is_empty() {
        path.push(' ');
        path.push_str(c.ext);
    }
    part(&mut out, "path", &path);
    part(&mut out, "key", head(c.name.unwrap_or(""), NAME_MAX));
    if c.value.chars().count() <= VALUE_MAX {
        part(&mut out, "value", c.value);
    } else {
        let mut v = String::with_capacity(VALUE_MAX * 4 + ELISION_MARK.len());
        v.push_str(head(c.value, VALUE_SIDE));
        v.push_str(ELISION_MARK);
        v.push_str(tail(c.value, VALUE_SIDE));
        part(&mut out, "value", &v);
    }
    out.push('\n');
    f.write(&mut out);
    part(
        &mut out,
        "line",
        &line_window(c.line_text, c.value_start, c.value_end),
    );
    part(
        &mut out,
        "prev",
        head(c.prev_line.unwrap_or(""), CONTEXT_MAX),
    );
    part(
        &mut out,
        "next",
        head(c.next_line.unwrap_or(""), CONTEXT_MAX),
    );
    out
}

/// The state of a candidate: its features ([`Features::of`]), then [`state_with`].
pub fn state(c: &Candidate<'_>, tok: &Tokenizer, words: &WordSet) -> String {
    state_with(c, &Features::of(c.value, tok, words, c.filters))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_scope_matches_creddata_tests() {
        // test_download_data.py::test_get_file_scope
        for (path, scope) in [
            ("/usr/local/lib/python", "/local/lib/python/usr/"),
            ("config/config.yml", "/conf/"),
            ("pet/modules/stdlib/spec/functions/t.rb", "/spec/modul/"),
            (
                "src/test/resources/dockerConfig/myDockerCfg",
                "/test/src/conf/resource/docker/",
            ),
            ("testapi/src/java/TestApiExample.java", "/test/src/example/"),
            (
                "packages/src/lib/__tests__/fixtures/request.json",
                "/test/src/fixture/lib/",
            ),
            ("X3", "/_/"),
        ] {
            assert_eq!(file_scope(path), scope, "{path}");
        }
        let big = WORD_IN_PATH.join("/");
        let scope = file_scope(&big);
        assert!(WORD_IN_PATH.iter().all(|w| scope.contains(w)), "{scope}");
        for w in WORD_IN_PATH {
            assert!(!file_scope(&format!("/X3/{w}")).contains("/_/"), "{w}");
        }
    }

    #[test]
    fn scope_words_are_the_directories_of_file_scope() {
        for path in [
            "src/test/resources/dockerConfig/myDockerCfg",
            "/usr/local/lib/python",
            "docs/doc/document/other/x",
            "X3",
            "",
        ] {
            let scope = file_scope(path);
            let dirs: Vec<&str> = scope.split('/').filter(|d| !d.is_empty()).collect();
            let words = scope_words(path);
            if words.is_empty() {
                assert_eq!(dirs, [NO_SCOPE], "{path}");
            } else {
                assert_eq!(dirs, words, "{path}");
            }
            // CredData files the copy under those directories. Reading the chain back as a
            // path gives the same chain here, but not always: see below
            if !words.is_empty() {
                let stored = format!("{}/0a1b2c3d", words.join("/"));
                assert_eq!(scope_words(&stored), words, "{path}");
            }
        }
    }

    #[test]
    fn a_stored_chain_is_not_always_its_own_scope() {
        // `another` holds `other`; filed under `other/`, the stored path also holds `/other`,
        // the second of the two words of the list that give that directory. The scope of a
        // CredData file is therefore read from its directories, never computed from them.
        assert_eq!(file_scope("x1/another/y"), "/other/");
        assert_eq!(scope_words("x1/another/y"), ["other"]);
        assert_eq!(scope_words("other/0a1b2c3d"), ["other", "other"]);
        assert_eq!(file_scope("a/other/x"), "/other/other/");
    }

    #[test]
    fn extension_as_python_splitext() {
        for (path, stem, ext) in [
            ("a/b/c.PY", "a/b/c", ".py"),
            ("a/b/.env", "a/b/.env", ""),
            ("a.b/c", "a.b/c", ""),
            ("x.tar.gz", "x.tar", ".gz"),
            ("..hidden.cfg", "..hidden", ".cfg"),
            ("Makefile", "Makefile", ""),
            ("dir/name.", "dir/name", "."),
            ("", "", ""),
        ] {
            assert_eq!(split_extension(path), (stem, ext.to_string()), "{path}");
        }
    }

    #[test]
    fn entropy_and_its_rounding() {
        assert_eq!(entropy(""), 0.0);
        assert_eq!(entropy("aaaa"), 0.0);
        assert_eq!(entropy("abab"), 1.0);
        // counts 8, 4, 2, 1, 1 of 16: 1.875 bits exactly, a tie at the hundredth; half up
        let v = "aaaaaaaabbbbccde";
        assert_eq!(entropy(v), 1.875);
        assert_eq!(hundredths(entropy(v)), 188);
        // counts 8, 2, 2, 2, 1, 1 of 16: 2.125
        assert_eq!(hundredths(entropy("aaaaaaaabbccddef")), 213);
        let mut s = String::new();
        write_hundredths(&mut s, 213);
        write_hundredths(&mut s, 5);
        assert_eq!(s, "2.130.05");
        // a character outside the BMP counts once and weighs two units of length: one distinct
        // character seen twice in four units is p = 1/2, as entropy.ts gives
        assert_eq!(entropy("\u{1F511}\u{1F511}"), 0.5);
        let h = entropy("\u{1F511}a");
        let want = -(1.0 / 3.0f64) * (1.0 / 3.0f64).log2() * 2.0;
        assert!((h - want).abs() < 1e-12, "{h}");
    }

    #[test]
    fn word_runs() {
        let words =
            WordSet::from_text("tree\nhouse\nTom\nabc\nhorse\nhouse\nverylongwordxx\ncafé\n");
        assert_eq!(words.len(), 3);
        assert_eq!(words.iter().collect::<Vec<_>>(), ["tree", "horse", "house"]);
        assert_eq!(words.of_len(5).len(), 2);
        assert!(words.of_len(6).is_empty());
        assert!(words.is_word("House") && words.is_word("TREE") && !words.is_word("hous"));
        assert_eq!(longest_word_run("Tree42!house", &words), 5);
        // a run is taken whole: two words glued together are not a word
        assert_eq!(longest_word_run("treehouse", &words), 0);
        assert_eq!(longest_word_run("x_tree_9", &words), 4);
        assert_eq!(longest_word_run("0123", &words), 0);
        assert_eq!(
            letter_runs("ab1cd--e").collect::<Vec<_>>(),
            [(0, 2), (3, 5), (7, 8)]
        );
    }

    #[test]
    fn the_line_window_is_centred_and_bounded() {
        let left = "L".repeat(200);
        let right = "R".repeat(200);
        let line = format!("{left}VALUE{right}");
        let w = line_window(&line, 200, 205);
        assert_eq!(w.chars().count(), LINE_MAX);
        assert_eq!(
            w,
            format!("{}{VALUE_MARK}{}", "L".repeat(58), "R".repeat(59))
        );
        // a short side gives its room to the other
        let line = format!("ab=VALUE{right}");
        let w = line_window(&line, 3, 8);
        assert_eq!(w, format!("ab={VALUE_MARK}{}", "R".repeat(114)));
        let line = format!("{left}VALUE;");
        let w = line_window(&line, 200, 205);
        assert_eq!(w, format!("{}{VALUE_MARK};", "L".repeat(116)));
        // a value that runs past its line, and columns past the end
        assert_eq!(
            line_window("key: abc", 5, 999),
            format!("key: {VALUE_MARK}")
        );
        assert_eq!(line_window("key", 10, 12), format!("key{VALUE_MARK}"));
        // columns are UTF-16 units: the key emoji takes two
        assert_eq!(
            line_window("\u{1F511}=xy;", 3, 5),
            format!("\u{1F511}={VALUE_MARK};")
        );
    }
}
