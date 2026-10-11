//! Hand-written GPT-2 pre-tokenizer split, equivalent to the Oniguruma regex
//! `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
//! (leftmost-first alternation, greedy quantifiers). `\s` is the Unicode White_Space property.

use super::unicode_tables::{LETTER, NUMBER};

fn in_table(table: &[(u32, u32)], c: char) -> bool {
    let cp = c as u32;
    table
        .binary_search_by(|&(lo, hi)| {
            if hi < cp {
                std::cmp::Ordering::Less
            } else if lo > cp {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// `\p{L}`
#[inline]
pub fn is_letter(c: char) -> bool {
    if c.is_ascii() {
        c.is_ascii_alphabetic()
    } else {
        in_table(LETTER, c)
    }
}

/// `\p{N}`
#[inline]
pub fn is_number(c: char) -> bool {
    if c.is_ascii() {
        c.is_ascii_digit()
    } else {
        in_table(NUMBER, c)
    }
}

#[inline]
fn is_other(c: char) -> bool {
    !c.is_whitespace() && !is_letter(c) && !is_number(c)
}

/// Iterator over the pre-tokens of `s`.
pub fn gpt2_split(s: &str) -> Gpt2Split<'_> {
    Gpt2Split { s, pos: 0 }
}

pub struct Gpt2Split<'a> {
    s: &'a str,
    pos: usize,
}

impl<'a> Iterator for Gpt2Split<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        if self.pos >= self.s.len() {
            return None;
        }
        let start = self.pos;
        let end = start + match_len(&self.s[start..]);
        self.pos = end;
        Some(&self.s[start..end])
    }
}

/// Byte length of the run of chars satisfying `pred` at the start of `s`.
#[inline]
fn run_len(s: &str, pred: impl Fn(char) -> bool) -> usize {
    s.char_indices()
        .find(|&(_, c)| !pred(c))
        .map_or(s.len(), |(i, _)| i)
}

/// Length of the regex match at the start of a non-empty `s`.
fn match_len(s: &str) -> usize {
    let mut chars = s.chars();
    let c0 = chars.next().expect("non-empty");
    let c1 = chars.next();

    // 's|'t|'re|'ve|'m|'ll|'d
    if c0 == '\'' {
        let r = &s[1..];
        for suf in ["s", "t", "re", "ve", "m", "ll", "d"] {
            if r.starts_with(suf) {
                return 1 + suf.len();
            }
        }
    }

    // ` ?X+` for X in \p{L}, \p{N}, [^\s\p{L}\p{N}]: the optional space is tried first
    let classes: [fn(char) -> bool; 3] = [is_letter, is_number, is_other];
    for class in classes {
        if c0 == ' ' {
            if let Some(c) = c1 {
                if class(c) {
                    return 1 + run_len(&s[1..], class);
                }
            }
        }
        if class(c0) {
            return run_len(s, class);
        }
    }

    // \s+(?!\S) then \s+
    debug_assert!(c0.is_whitespace());
    let ws = run_len(s, char::is_whitespace);
    if ws == s.len() {
        return ws; // whitespace up to the end
    }
    // followed by a non-space: give back the last whitespace char if the run has more than one
    let last = s[..ws]
        .char_indices()
        .next_back()
        .map(|(i, _)| i)
        .unwrap_or(0);
    if last > 0 {
        last
    } else {
        ws
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(s: &str) -> Vec<&str> {
        gpt2_split(s).collect()
    }

    #[test]
    fn gpt2_regex_cases() {
        assert_eq!(split("Hello world"), vec!["Hello", " world"]);
        assert_eq!(split("I'm don't"), vec!["I", "'m", " don", "'t"]);
        assert_eq!(split("a   b"), vec!["a", "  ", " b"]);
        assert_eq!(split("x -3 12"), vec!["x", " -", "3", " 12"]);
        assert_eq!(split("end  "), vec!["end", "  "]);
        assert_eq!(split("a\n\nb"), vec!["a", "\n", "\n", "b"]);
        assert_eq!(split(" \tb"), vec![" ", "\t", "b"]);
        assert_eq!(split("'''re"), vec!["'''", "re"]);
        assert_eq!(split("café ½"), vec!["café", " ½"]);
        assert_eq!(split("{\"a\": 1}"), vec!["{\"", "a", "\":", " 1", "}"]);
        assert!(
            is_letter('é')
                && is_letter('日')
                && !is_letter('1')
                && is_number('½')
                && is_number('٣')
        );
    }
}
