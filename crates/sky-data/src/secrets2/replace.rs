//! The word-preserving shape replacement of the issue-report positives
//! (`docs/s6-secrets2-prereg.md`, section 3.2): a real value becomes an invented one of the same
//! shape before it enters a training or validation file.
//!
//! - **Known provider prefixes are kept**: the prefixes CredData's own obfuscation keeps
//!   (`obfuscate_creds.py`, `get_prefix()` and the cases of `get_obfuscated_value()`), and those
//!   of this repository's generator (`crate::secrets`). A prefix counts where a token starts: at
//!   the start of the value or after a character that is neither a letter nor a digit, since an
//!   issue-report candidate is a regular expression's match and often holds text before its key.
//! - **Separators are kept**: every character that is not an ASCII letter or digit.
//! - **A dictionary word stays a word**: a maximal run of ASCII letters of 4 characters or more
//!   that is a word of the dictionary becomes another word of the same length, in the same case
//!   pattern. This is what CredData's obfuscation does not do.
//! - **A context run is kept as it is** (amendments 4 and 5): a candidate of the issue-report
//!   set is a regular expression's match and often holds its key name (`token`, `password`,
//!   `api_key`), which would otherwise be rewritten in positives and never in negatives. A whole
//!   run of letters is left untouched when it occurs, as a whole run and without regard to
//!   case, in at least [`KEPT_MIN_STRINGS`] distinct negative candidate strings of the training
//!   split, and either it is a dictionary word, or it has at least [`KEPT_MIN_LETTERS`] letters
//!   and contains one of the generator's keywords or scheme words ([`CONTEXT_WORDS`]). The
//!   second case asks for a keyword so that a short random run of a real value is not kept
//!   because it happens to be common ([`KeptRuns`]).
//! - **Every other letter or digit becomes a random one of its class**, the classes being
//!   CredData's: a run of letters and digits written in lower-case hexadecimal stays so, as does
//!   upper-case hexadecimal and base32 (`A-Z`, `2-7`); otherwise a lower-case letter becomes a
//!   lower-case letter, an upper-case letter an upper-case letter, a digit a digit.
//! - **A JSON Web Token stays one**: a part that starts with `eyJ` is decoded, its JSON keeps
//!   its structure and the claim names CredData keeps, and is encoded again at the same length.
//!
//! The length never changes, character for character, so offsets into the text stay valid. The
//! draw depends on the seed and on the value alone: the same value gets the same replacement
//! wherever it stands.

use std::collections::{HashMap, HashSet};

use sky_infer::secret_cand::{letter_runs, WordSet};

use super::sha256_hex;
use crate::rng::Rng;

/// Distinct negative candidate strings a run must occur in, as a whole run of letters, to be
/// kept as it is (amendment 4).
pub const KEPT_MIN_STRINGS: usize = 5;

/// Letters a run that is no dictionary word must have to be kept (amendment 5).
pub const KEPT_MIN_LETTERS: usize = 3;

/// The keywords of the candidate generator (`generic-keyword`) and its scheme words: a run that
/// is no dictionary word is kept only if it holds one of them (amendment 5).
pub const CONTEXT_WORDS: [&str; 18] = [
    "password",
    "passwd",
    "pwd",
    "pass",
    "secret",
    "token",
    "key",
    "api",
    "auth",
    "credential",
    "cred",
    "nonce",
    "salt",
    "bearer",
    "basic",
    "digest",
    "negotiate",
    "apikey",
];

/// Whether a run of letters may be kept if the negatives hold it often enough: a dictionary
/// word, or a run of at least [`KEPT_MIN_LETTERS`] letters that contains a context word.
pub fn may_be_kept(run: &str, words: &WordSet) -> bool {
    if words.is_word(run) {
        return true;
    }
    let low = run.to_ascii_lowercase();
    low.len() >= KEPT_MIN_LETTERS && CONTEXT_WORDS.iter().any(|w| low.contains(w))
}

/// The runs of letters that are kept as they are, in lower case, and for the counts every run
/// of at least [`KEPT_MIN_LETTERS`] letters that the negatives hold as often.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KeptRuns {
    runs: HashSet<String>,
    common: HashSet<String>,
}

impl KeptRuns {
    /// From the negative candidate strings of the training split (after its decontamination):
    /// each distinct string counts once for each run of letters it holds, whatever the case of
    /// the run. A run of at least [`KEPT_MIN_STRINGS`] strings is kept if [`may_be_kept`].
    pub fn from_negatives<'a>(
        negatives: impl IntoIterator<Item = &'a str>,
        words: &WordSet,
    ) -> Self {
        let distinct: HashSet<&str> = negatives.into_iter().collect();
        let mut strings: HashMap<String, usize> = HashMap::new();
        for s in distinct {
            let runs: HashSet<String> = letter_runs(s)
                .map(|(a, b)| s[a..b].to_ascii_lowercase())
                .filter(|run| run.len() >= KEPT_MIN_LETTERS)
                .collect();
            for run in runs {
                *strings.entry(run).or_default() += 1;
            }
        }
        let common: HashSet<String> = strings
            .into_iter()
            .filter(|(_, n)| *n >= KEPT_MIN_STRINGS)
            .map(|(run, _)| run)
            .collect();
        Self {
            runs: common
                .iter()
                .filter(|run| may_be_kept(run, words))
                .cloned()
                .collect(),
            common,
        }
    }

    /// Whether a run of letters is kept, without regard to its case.
    pub fn contains(&self, run: &str) -> bool {
        !self.runs.is_empty() && self.runs.contains(&run.to_ascii_lowercase())
    }

    /// Whether at least [`KEPT_MIN_STRINGS`] negatives hold the run, kept or not.
    pub fn is_common(&self, run: &str) -> bool {
        !self.common.is_empty() && self.common.contains(&run.to_ascii_lowercase())
    }

    pub fn len(&self) -> usize {
        self.runs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }
}

/// A value's replacement, and what the word rule did in it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Replacement {
    pub text: String,
    /// The runs of letters kept as they are, in lower case, one entry per run.
    pub kept: Vec<String>,
    /// Dictionary-word runs replaced by another word.
    pub replaced: usize,
    /// Runs that the negatives hold as often as a kept run and that are drawn again all the
    /// same: no dictionary word, no keyword in them.
    pub redrawn_common: usize,
}

/// Prefixes kept where a token starts.
const PREFIXES: [&str; 86] = [
    // CredData, get_prefix()
    "k1.secret.",
    "k2.secret.",
    "k3.secret.",
    "k4.secret.",
    "k1.secret-pw.",
    "k2.secret-pw.",
    "k3.secret-pw.",
    "k4.secret-pw.",
    "k1.local.",
    "k2.local.",
    "k3.local.",
    "k4.local.",
    "k1.local-pw.",
    "k2.local-pw.",
    "k3.local-pw.",
    "k4.local-pw.",
    "v1.local.",
    "v2.local.",
    "v3.local.",
    "v4.local.",
    "v1.public.",
    "v2.public.",
    "v3.public.",
    "v4.public.",
    "TlRMTVNTUAABAAAA",
    "TlRMTVNTUAACAAAA",
    "TlRMTVNTUAADAAAA",
    "whsec_",
    "cosmo_",
    "OAuth ",
    "hexkey:",
    "base64:",
    "phpass:",
    "Bearer ",
    "Apikey ",
    "hexpass:",
    "hexsalt:",
    "pk_live_",
    "rk_live_",
    "sk_live_",
    "pk_test_",
    "rk_test_",
    "sk_test_",
    "SWMTKN-1-",
    "dckr_pat_",
    "dckr_oat_",
    "hexsecret:",
    "sk-ant-api03-",
    "AKIA",
    "ABIA",
    "ACCA",
    "AGPA",
    "AIDA",
    "AIPA",
    "ANPA",
    "ANVA",
    "AROA",
    "APKA",
    "ASCA",
    "ASIA",
    "AIza",
    "AKGP",
    "LTAI",
    "ya29.",
    "pass:",
    "salt:",
    "akab-",
    "PMAK-",
    "PMAT-",
    "xapp-",
    "pplx-",
    "hooks.slack.com/services/",
    // CredData, get_obfuscated_value(): the first characters it keeps
    "1//0",
    "phc_",
    "key-",
    "squ_",
    "glsa_",
    "ghr_",
    // this repository's generator (and CredData's checksummed tokens)
    "ghp_",
    "gho_",
    "ghu_",
    "ghs_",
    "npm_",
    "github_pat_",
    "glpat-",
    "sk-proj-",
];

/// Shorter prefixes of the same families, tried after [`PREFIXES`].
const SHORT_PREFIXES: [&str; 3] = ["hf_", "SG.", "sk-"];

/// Text CredData keeps inside a value.
const KEPT_INSIDE: [&str; 8] = [
    ".apps.googleusercontent.com",
    ".s3.amazonaws.com",
    ".firebaseio.com",
    ".firebaseapp.com",
    ":ED25519:",
    ":X25519:",
    ":ED448:",
    ":X448:",
];

/// Two-letter prefixes of 34-character identifiers that CredData keeps.
const ID34: [&str; 25] = [
    "AC", "AD", "AL", "CA", "CF", "CL", "CN", "CR", "FW", "IP", "KS", "MM", "NO", "PK", "PN", "QU",
    "RE", "SC", "SD", "SK", "SM", "TR", "UT", "XE", "XR",
];

/// JSON words and JWT claim names that CredData's `obfuscate_jwt()` keeps.
const JWT_KEPT: [&str; 60] = [
    "null",
    "false",
    "true",
    "\"alg\":",
    "\"apu\":",
    "\"apv\":",
    "\"aud\":",
    "\"b64\":",
    "\"crit\":",
    "\"crv\":",
    "\"cty\":",
    "\"d\":",
    "\"dp\":",
    "\"dq\":",
    "\"e\":",
    "\"enc\":",
    "\"epk\":",
    "\"exp\":",
    "\"ext\":",
    "\"iat\":",
    "\"id\":",
    "\"iss\":",
    "\"iv\":",
    "\"jku\":",
    "\"jti\":",
    "\"jwk\":",
    "\"k\":",
    "\"key_ops\":",
    "\"keys\":",
    "\"kid\":",
    "\"kty\":",
    "\"n\":",
    "\"nbf\":",
    "\"nonce\":",
    "\"oth\":",
    "\"p\":",
    "\"p2c\":",
    "\"p2s\":",
    "\"password\":",
    "\"ppt\":",
    "\"q\":",
    "\"qi\":",
    "\"role\":",
    "\"secret\":",
    "\"sub\":",
    "\"svt\":",
    "\"tag\":",
    "\"token\":",
    "\"typ\":",
    "\"url\":",
    "\"use\":",
    "\"x\":",
    "\"x5c\":",
    "\"x5t\":",
    "\"x5t#S256\":",
    "\"x5u\":",
    "\"y\":",
    "\"zip\":",
    "\"o\":",
    "\"m\":",
];

const B64_STD: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const B64_URL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

fn b64_decode(s: &[u8]) -> Option<Vec<u8>> {
    let val = |c: u8| -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => return None,
        } as u32)
    };
    if s.len() % 4 == 1 {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    for chunk in s.chunks(4) {
        let mut acc = 0u32;
        for &c in chunk {
            acc = (acc << 6) | val(c)?;
        }
        acc <<= 6 * (4 - chunk.len());
        let bytes = acc.to_be_bytes();
        out.extend_from_slice(&bytes[1..chunk.len()]);
    }
    Some(out)
}

fn b64_encode(bytes: &[u8], alphabet: &[u8; 64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let mut acc = 0u32;
        for (i, &b) in chunk.iter().enumerate() {
            acc |= (b as u32) << (16 - 8 * i);
        }
        for i in 0..=chunk.len() {
            out.push(alphabet[((acc >> (18 - 6 * i)) & 63) as usize]);
        }
    }
    out
}

fn random_of(rng: &mut Rng, set: &[u8]) -> u8 {
    set[rng.below(set.len())]
}

const LOWER: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
const UPPER: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const DIGIT: &[u8] = b"0123456789";

/// A JWT part (`eyJ…`, base64 of a JSON object) with its content replaced: structure, reserved
/// words and claim names kept, every other letter and digit drawn again in its class. `None`
/// when the part does not decode.
fn replace_jwt_part(part: &[u8], rng: &mut Rng) -> Option<Vec<u8>> {
    let decoded = b64_decode(part)?;
    // `eyJ` is `{"` and the first bits of a third byte: without that byte there is no object
    if decoded.len() < 3 || !decoded.starts_with(b"{\"") {
        return None;
    }
    let mut out = decoded.clone();
    let mut n = 0;
    while n < decoded.len() {
        let c = decoded[n];
        if matches!(c, b'n' | b'f' | b't' | b'"') {
            if let Some(w) = JWT_KEPT
                .iter()
                .find(|w| decoded[n..].starts_with(w.as_bytes()))
            {
                n += w.len();
                continue;
            }
        }
        out[n] = match c {
            b'0'..=b'9' => random_of(rng, &DIGIT[1..]),
            b'a'..=b'z' => random_of(rng, LOWER),
            b'A'..=b'Z' => random_of(rng, UPPER),
            b'\\' => {
                // an escape and the character it escapes
                if n + 1 < decoded.len() {
                    out[n + 1] = b'?';
                    n += 1;
                }
                b'?'
            }
            _ => c,
        };
        n += 1;
    }
    let url = part.iter().any(|&c| c == b'-' || c == b'_')
        || !part.iter().any(|&c| c == b'+' || c == b'/');
    let mut enc = b64_encode(&out, if url { B64_URL } else { B64_STD });
    enc.truncate(part.len());
    // a part stays a JWT part whatever stood after its first two bytes
    (enc.len() == part.len() && enc.starts_with(b"ey")).then(|| {
        enc[2] = b'J';
        enc
    })
}

fn is_alnum(c: u8) -> bool {
    c.is_ascii_alphanumeric()
}

/// The replacement of `value`, drawn from `rng`. Works on bytes: only ASCII letters and digits
/// change, so every other character (and every multi-byte one) is copied.
fn replace_with(value: &str, words: &WordSet, keep: &KeptRuns, rng: &mut Rng) -> Replacement {
    let mut report = Replacement::default();
    let b = value.as_bytes();
    let n = b.len();
    let mut out = b.to_vec();
    // kept[i]: the character is copied as it is; done[i]: it has its replacement
    let mut kept = vec![false; n];
    let mut done = vec![false; n];
    for inside in KEPT_INSIDE {
        let mut from = 0;
        while let Some(i) = value[from..].find(inside) {
            let at = from + i;
            kept[at..at + inside.len()].fill(true);
            from = at + inside.len();
        }
    }
    let mut i = 0;
    while i < n {
        let token_start = i == 0 || !is_alnum(b[i - 1]);
        if !token_start || !is_alnum(b[i]) || kept[i] {
            i += 1;
            continue;
        }
        let rest = &value[i..];
        // the token: the run of letters and digits that starts here
        let run = rest.bytes().take_while(|&c| is_alnum(c)).count();
        if rest.starts_with("eyJ") {
            // a JWT: parts of base64 characters between dots
            let end = i + rest
                .bytes()
                .take_while(|&c| is_alnum(c) || matches!(c, b'-' | b'_' | b'.' | b'+' | b'/'))
                .count();
            let mut at = i;
            for part in value[i..end].split('.') {
                if part.starts_with("eyJ") {
                    if let Some(new) = replace_jwt_part(part.as_bytes(), rng) {
                        out[at..at + part.len()].copy_from_slice(&new);
                        done[at..at + part.len()].fill(true);
                    } else {
                        kept[at..at + 3].fill(true);
                    }
                }
                at += part.len() + 1;
            }
            i = end;
            continue;
        }
        let prefix = PREFIXES
            .iter()
            .chain(&SHORT_PREFIXES)
            .find(|p| rest.starts_with(**p))
            .map(|p| p.len())
            .or_else(|| {
                let tok = &rest[..run];
                let slack = rest.len() >= 15
                    && rest.starts_with("xox")
                    && b"abeoprst".contains(&rest.as_bytes()[3])
                    && rest.as_bytes()[4] == b'-';
                if slack {
                    Some(4)
                } else if (run == 34 && ID34.iter().any(|p| tok.starts_with(p)))
                    || (run == 18 && tok.starts_with("wx"))
                {
                    Some(2)
                } else if (12..=18).contains(&run) && tok.starts_with("00D") {
                    Some(3)
                } else {
                    None
                }
            });
        match prefix {
            Some(p) => {
                kept[i..i + p].fill(true);
                i += p;
            }
            None => i += run.max(1),
        }
    }
    // dictionary words: maximal runs of letters, cut where a kept part starts or ends
    for (a, e) in letter_runs(value) {
        let mut s = a;
        while s < e {
            if kept[s] || done[s] {
                s += 1;
                continue;
            }
            let mut t = s;
            while t < e && !kept[t] && !done[t] {
                t += 1;
            }
            let run = &value[s..t];
            if keep.contains(run) {
                // a context run: not a character of it changes
                kept[s..t].fill(true);
                report.kept.push(run.to_ascii_lowercase());
            } else if !words.is_word(run) {
                report.redrawn_common += keep.is_common(run) as usize;
            } else {
                let pool = words.of_len(run.len());
                let lower = run.to_ascii_lowercase();
                if pool.len() > 1 {
                    report.replaced += 1;
                    let new = loop {
                        let w = &pool[rng.below(pool.len())];
                        if **w != *lower {
                            break w;
                        }
                    };
                    for (k, c) in new.bytes().enumerate() {
                        out[s + k] = if b[s + k].is_ascii_uppercase() {
                            c.to_ascii_uppercase()
                        } else {
                            c
                        };
                    }
                    done[s..t].fill(true);
                }
            }
            s = t;
        }
    }
    // every other letter and digit, by runs of letters and digits: the class of the run
    let mut i = 0;
    while i < n {
        if !is_alnum(b[i]) || kept[i] || done[i] {
            i += 1;
            continue;
        }
        let mut t = i;
        while t < n && is_alnum(b[t]) && !kept[t] && !done[t] {
            t += 1;
        }
        let run = &b[i..t];
        let hex_lower = run.iter().all(|c| matches!(c, b'0'..=b'9' | b'a'..=b'f'));
        let hex_upper = run.iter().all(|c| matches!(c, b'0'..=b'9' | b'A'..=b'F'));
        let base32 = run.iter().all(|c| matches!(c, b'A'..=b'Z' | b'2'..=b'7'));
        let (lower, upper, digit): (&[u8], &[u8], &[u8]) = if hex_lower {
            (&LOWER[..6], UPPER, DIGIT)
        } else if hex_upper {
            (LOWER, &UPPER[..6], DIGIT)
        } else if base32 {
            (LOWER, UPPER, &DIGIT[2..8])
        } else {
            (LOWER, UPPER, DIGIT)
        };
        for k in i..t {
            out[k] = match b[k] {
                b'0'..=b'9' => random_of(rng, digit),
                b'a'..=b'z' => random_of(rng, lower),
                _ => random_of(rng, upper),
            };
        }
        i = t;
    }
    // only ASCII bytes were written over ASCII bytes
    report.text = String::from_utf8(out).unwrap_or_else(|_| value.to_string());
    report
}

/// The replacement of `value`: the same for the same value, kept runs and seed. A value that
/// has a letter or a digit to draw never comes back unchanged (it is drawn again, a few times
/// at most); a value made of kept runs and separators only has nothing to draw and stays.
pub fn replace_value_report(
    value: &str,
    words: &WordSet,
    keep: &KeptRuns,
    seed: u64,
) -> Replacement {
    let mut rng = Rng::derived(
        seed,
        &format!("s6-secrets2:replace:{}", sha256_hex(value.as_bytes())),
    );
    let mut out = replace_with(value, words, keep, &mut rng);
    for _ in 0..16 {
        if out.text != value {
            break;
        }
        out = replace_with(value, words, keep, &mut rng);
    }
    out
}

/// The text of [`replace_value_report`].
pub fn replace_value(value: &str, words: &WordSet, keep: &KeptRuns, seed: u64) -> String {
    replace_value_report(value, words, keep, seed).text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secrets::{rand_str, ALNUM, HEX, UPPER_DIGITS, URLSAFE};

    fn none() -> KeptRuns {
        KeptRuns::default()
    }

    fn words() -> WordSet {
        WordSet::from_text(
            "tree\nbird\nlamp\nhouse\nhorse\nmouse\nwinter\nsummer\nbattery\ncabinet\npassword\nsandwich\n",
        )
    }

    /// The shape of a value: `a` for a lower-case letter, `A` for an upper-case one, `0` for a
    /// digit, anything else as it is.
    fn shape(s: &str) -> String {
        s.chars()
            .map(|c| match c {
                'a'..='z' => 'a',
                'A'..='Z' => 'A',
                '0'..='9' => '0',
                c => c,
            })
            .collect()
    }

    #[test]
    fn shape_and_length_are_kept_and_the_draw_is_stable() {
        let w = words();
        let mut rng = Rng::derived(3, "replace-test");
        for _ in 0..300 {
            let v = format!(
                "{}-{}_{}",
                rand_str(&mut rng, ALNUM, 12),
                rand_str(&mut rng, UPPER_DIGITS, 9),
                rand_str(&mut rng, URLSAFE, 20)
            );
            let r = replace_value(&v, &w, &none(), 1234);
            assert_eq!(shape(&r), shape(&v));
            assert_eq!(r.len(), v.len());
            assert_ne!(r, v);
            assert_eq!(r, replace_value(&v, &w, &none(), 1234));
            assert_ne!(r, replace_value(&v, &w, &none(), 1235));
        }
    }

    #[test]
    fn a_word_stays_a_word_in_its_case() {
        let w = words();
        for (v, word_at, len) in [
            ("Winter2024!", 0, 6),
            ("my-HORSE-77", 3, 5),
            ("x9_battery_x9", 3, 7),
        ] {
            let r = replace_value(v, &w, &none(), 7);
            let new = &r[word_at..word_at + len];
            let old = &v[word_at..word_at + len];
            assert!(w.is_word(new), "{new}");
            assert_ne!(new.to_ascii_lowercase(), old.to_ascii_lowercase());
            assert_eq!(shape(new), shape(old));
            assert_eq!(shape(&r), shape(v));
        }
        // glued words are one run, which is no word: letters of its class
        let r = replace_value("treehouse", &w, &none(), 7);
        assert_eq!(shape(&r), "aaaaaaaaa");
        // a word with no other word of its length cannot be replaced by a word
        let alone = WordSet::from_text("tree\n");
        assert_eq!(
            shape(&replace_value("tree-12", &alone, &none(), 7)),
            "aaaa-00"
        );
        assert_ne!(replace_value("tree-12", &alone, &none(), 7), "tree-12");
    }

    #[test]
    fn a_context_word_of_the_negatives_is_kept_as_it_is() {
        let w = words();
        // `password` stands as a whole run in five distinct negatives, in several cases;
        // `winter` in four; `house` in five strings of which two are the same string
        let negatives = [
            "password: none",
            "PASSWORD=",
            "db_password = x1",
            "Password required",
            "the password, again",
            "winter 1",
            "Winter 2",
            "winter-3",
            "WINTER_4",
            "house a",
            "house b",
            "house c",
            "house d",
            "house d",
            // glued to other letters it is another run, which counts for nothing
            "passwords mypassword housekeeping",
        ];
        let keep = KeptRuns::from_negatives(negatives, &w);
        assert!(keep.contains("password") && keep.contains("PassWord"));
        assert!(!keep.contains("winter") && !keep.contains("house") && !keep.contains("none"));
        assert_eq!(keep.len(), 1);
        // a fifth distinct string, and the word is kept
        let more = negatives.iter().copied().chain(["house e"]);
        assert!(KeptRuns::from_negatives(more, &w).contains("house"));

        let secret = "Password: Winter-48213-qzv";
        let r = replace_value_report(secret, &w, &keep, 7);
        // the key name is untouched, in its case; the word of the value is still replaced
        assert!(r.text.starts_with("Password: "), "the context word stays");
        assert_eq!(
            (r.kept.clone(), r.replaced),
            (vec!["password".to_string()], 1)
        );
        let word = &r.text[10..16];
        assert!(
            w.is_word(word)
                && word != "Winter"
                && word.starts_with(|c: char| c.is_ascii_uppercase())
        );
        assert_eq!(shape(&r.text), shape(secret));
        assert!(!r.text.contains("48213") || !r.text.contains("qzv"));
        // without the kept runs, the rule of section 3.2 as first written: both words go
        let before = replace_value_report(secret, &w, &none(), 7);
        assert!(!before.text.starts_with("Password"));
        assert_eq!((before.kept.len(), before.replaced), (0, 2));
        // a kept word is kept only as a whole run
        let r = replace_value_report("mypassword=Summer77", &w, &keep, 7);
        assert!(!r.text.starts_with("mypassword") && r.kept.is_empty());
        // a value made of kept words and separators has nothing to draw
        let r = replace_value_report("password: PASSWORD", &w, &keep, 7);
        assert_eq!(
            (r.text.as_str(), r.kept.len(), r.replaced),
            ("password: PASSWORD", 2, 0)
        );
    }

    #[test]
    fn a_key_name_that_is_no_dictionary_word_is_kept_when_it_holds_a_keyword() {
        let w = words();
        // amendment 5: five distinct negatives each hold `api`, `key`, `apikey`, `pwd`,
        // `accesstoken`, `com` and `id` as whole runs; four hold `authz`
        let negatives: Vec<String> = (0..5)
            .map(|i| {
                format!(
                    "api_key={i} apikey pwd accessToken example.com id {}",
                    if i < 4 { "authz" } else { "" }
                )
            })
            .collect();
        let keep = KeptRuns::from_negatives(negatives.iter().map(String::as_str), &w);
        for run in ["api", "KEY", "apikey", "ApiKey", "pwd", "accesstoken"] {
            assert!(keep.contains(run), "{run}");
        }
        // common, but no dictionary word and no keyword in it; or too short; or too rare
        assert!(keep.is_common("com") && !keep.contains("com"));
        assert!(keep.is_common("example") && !keep.contains("example"));
        assert!(!keep.contains("id") && !keep.is_common("id"));
        assert!(!keep.contains("authz") && !keep.is_common("authz"));
        assert_eq!(keep.len(), 5);
        assert!(
            may_be_kept("passwd", &w) && may_be_kept("XApiKey", &w) && may_be_kept("winter", &w)
        );
        assert!(!may_be_kept("ke", &w) && !may_be_kept("qzv", &w) && !may_be_kept("mongodb", &w));

        let secret = "api_key=Qz81mfa22Lx; apikey: Winter-77 (pwd) example.com authz";
        let r = replace_value_report(secret, &w, &keep, 7);
        assert!(
            r.text.starts_with("api_key=")
                && r.text.contains("; apikey: ")
                && r.text.contains(" (pwd) ")
        );
        assert_eq!(r.kept, ["api", "key", "apikey", "pwd"]);
        // the value is still drawn again, its word replaced by a word; `com`, `example` and
        // `authz` are drawn again too, the first two although the negatives hold them
        assert!(!r.text.contains("Qz81mfa22Lx") && !r.text.contains("Winter"));
        assert!(!r.text.contains("example") && !r.text.ends_with("authz"));
        assert_eq!((r.replaced, r.redrawn_common), (1, 2));
        assert_eq!(shape(&r.text), shape(secret));
        // a run is kept as a whole run only: `mykey` is not `key`
        let r = replace_value_report("mykey=Qz81", &w, &keep, 7);
        assert!(!r.text.starts_with("mykey") && r.kept.is_empty());
        // with amendment 4 alone (dictionary words only) these names were drawn again
        let before = KeptRuns::from_negatives(["winter 1"], &w);
        assert!(!replace_value("api_key=Qz81", &w, &before, 7).starts_with("api_key"));
    }

    #[test]
    fn classes_follow_creddata() {
        let w = words();
        let mut rng = Rng::derived(5, "replace-classes");
        for _ in 0..200 {
            let hex = rand_str(&mut rng, HEX, 40);
            let r = replace_value(&hex, &w, &none(), 1);
            assert!(
                r.bytes().all(|c| HEX.contains(&c)),
                "lower hex stays lower hex"
            );
            let upper = hex.to_ascii_uppercase();
            let r = replace_value(&upper, &w, &none(), 1);
            assert!(r.bytes().all(|c| b"0123456789ABCDEF".contains(&c)));
            let b32 = rand_str(&mut rng, b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567", 32);
            let r = replace_value(&b32, &w, &none(), 1);
            assert!(r
                .bytes()
                .all(|c| b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567".contains(&c)));
        }
        // a UUID keeps its dashes and its hexadecimal groups
        let uuid = format!(
            "{}-{}-{}-{}-{}",
            rand_str(&mut rng, HEX, 8),
            rand_str(&mut rng, HEX, 4),
            rand_str(&mut rng, HEX, 4),
            rand_str(&mut rng, HEX, 4),
            rand_str(&mut rng, HEX, 12)
        );
        let r = replace_value(&uuid, &w, &none(), 1);
        assert!(r.bytes().all(|c| c == b'-' || HEX.contains(&c)));
        assert_eq!(shape(&r).matches('-').count(), 4);
        // what is not an ASCII letter or digit is copied
        let v = "p\u{e4}ss w\u{f6}rd:\t#12 \u{1F512}";
        let r = replace_value(v, &w, &none(), 1);
        assert_eq!(shape(&r), shape(v));
        assert_eq!(r.chars().count(), v.chars().count());
    }

    #[test]
    fn provider_prefixes_are_kept_where_a_token_starts() {
        let w = words();
        let mut rng = Rng::derived(9, "replace-prefixes");
        // prefixes assembled here, with invented bodies
        let cases = [
            (
                format!(
                    "{}{}",
                    ["AK", "IA"].concat(),
                    rand_str(&mut rng, b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567", 16)
                ),
                4,
            ),
            (
                format!("{}_{}", ["gh", "p"].concat(), rand_str(&mut rng, ALNUM, 36)),
                4,
            ),
            (
                format!(
                    "{}_{}",
                    ["sk", "live"].join("_"),
                    rand_str(&mut rng, ALNUM, 24)
                ),
                8,
            ),
            (
                format!(
                    "{}-{}",
                    ["xo", "xb"].concat(),
                    rand_str(&mut rng, ALNUM, 30)
                ),
                4,
            ),
            (
                format!(
                    "{}_{}",
                    ["github", "pat"].join("_"),
                    rand_str(&mut rng, ALNUM, 40)
                ),
                11,
            ),
            (
                format!("{}{}", ["AC"].concat(), rand_str(&mut rng, HEX, 32)),
                2,
            ),
        ];
        for (v, keep) in &cases {
            let r = replace_value(v, &w, &none(), 11);
            assert_eq!(
                &r[..*keep],
                &v[..*keep],
                "prefix of a {}-character value",
                v.len()
            );
            assert_ne!(&r[*keep..], &v[*keep..]);
            assert_eq!(shape(&r), shape(v));
            // after text, as in a regular expression's match
            let line = format!("token: \"{v}\"");
            let r = replace_value(&line, &w, &none(), 11);
            assert_eq!(&r[8..8 + keep], &v[..*keep]);
            assert_eq!(shape(&r), shape(&line));
        }
        // glued to a letter, the same characters are no prefix
        let body = rand_str(&mut rng, b"BCDEFGHJKLMNOPQRSTUVWXYZ", 16);
        let v = format!("x{}{body}", ["AK", "IA"].concat());
        let r = replace_value(&v, &w, &none(), 11);
        assert_eq!(shape(&r), shape(&v));
        // a kept domain
        let v = format!(
            "{}.s3.amazonaws.com/{}",
            rand_str(&mut rng, HEX, 12),
            rand_str(&mut rng, ALNUM, 8)
        );
        let r = replace_value(&v, &w, &none(), 11);
        assert!(r.contains(".s3.amazonaws.com/"));
        assert_ne!(r, v);
    }

    #[test]
    fn a_jwt_stays_a_jwt() {
        let w = words();
        let header = b64_encode(br#"{"alg":"HS256","typ":"JWT"}"#, B64_URL);
        let payload = b64_encode(
            br#"{"sub":"user-4821","name":"Jo Doe","admin":true,"iat":1516239022}"#,
            B64_URL,
        );
        let mut rng = Rng::derived(2, "replace-jwt");
        let sig = rand_str(&mut rng, URLSAFE, 43);
        let jwt = format!(
            "{}.{}.{sig}",
            String::from_utf8(header).unwrap(),
            String::from_utf8(payload).unwrap()
        );
        let r = replace_value(&jwt, &w, &none(), 4);
        assert_eq!(r.len(), jwt.len());
        assert_ne!(r, jwt);
        let parts: Vec<&str> = r.split('.').collect();
        assert_eq!(parts.len(), 3);
        let head = String::from_utf8(b64_decode(parts[0].as_bytes()).unwrap()).unwrap();
        assert!(
            head.starts_with("{\"alg\":\"") && head.contains("\"typ\":\""),
            "{head}"
        );
        let body = String::from_utf8(b64_decode(parts[1].as_bytes()).unwrap()).unwrap();
        assert!(
            body.starts_with("{\"sub\":\"") && body.contains("\"iat\":") && body.contains(":true,")
        );
        assert!(!body.contains("user-4821") && !body.contains("Jo Doe"));
        assert_ne!(parts[2], sig);
        // a part that does not decode keeps its first three characters only
        let broken = format!("eyJ{}", rand_str(&mut rng, b"!bcd", 9));
        assert!(replace_value(&broken, &w, &none(), 4).starts_with("eyJ"));
    }

    #[test]
    fn base64_round_trip() {
        for s in ["", "a", "ab", "abc", "abcd", "{\"k\":1}"] {
            let e = b64_encode(s.as_bytes(), B64_STD);
            assert_eq!(b64_decode(&e).unwrap(), s.as_bytes());
        }
        assert!(b64_decode(b"a").is_none() && b64_decode(b"ab!d").is_none());
    }
}
