//! Native-number extraction of the `skycmd-s2` models (SPEC-S2 §2), applied to every text
//! (instructions, option texts, state) before BPE.
//!
//! Rules, which `sky-infer` and `sky-train` mirror bit for bit
//! (test vectors: `tests/fixtures/num_extract.json`):
//! - matches of the regex `-?\d+(?:\.\d+)?` over ASCII digits, scanned left to right without
//!   overlap (leftmost, greedy, like a regex `find_iter`);
//! - a match whose preceding character is an ASCII letter (`b2`, `glm-5`, `x1.5`) is kept
//!   verbatim and scanning resumes after it (so the `5` of `glm-5` is not extracted either);
//! - every other match is replaced by [`NUM_TOKEN`] and its value, the matched text parsed as
//!   an `f32` (round to nearest, so `-0` gives `-0.0`), is appended to the values.
//!
//! Only `std` is used, on purpose: the function must be trivially portable.

/// Replacement text of an extracted number; a special token of the s2 tokenizers.
pub const NUM_TOKEN: &str = "[NUM]";

/// Fixed id of [`NUM_TOKEN`] in the s2 tokenizers.
pub const NUM_ID: u32 = 5;

/// End (exclusive byte index) of the match of `-?\d+(?:\.\d+)?` that starts at byte `i`.
fn match_end(b: &[u8], i: usize) -> Option<usize> {
    let mut j = i;
    if b[j] == b'-' {
        j += 1;
    }
    let digits = j;
    while j < b.len() && b[j].is_ascii_digit() {
        j += 1;
    }
    if j == digits {
        return None;
    }
    if j + 1 < b.len() && b[j] == b'.' && b[j + 1].is_ascii_digit() {
        j += 2;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
    }
    Some(j)
}

/// Replaces every extracted number of `text` by `[NUM]` and returns the new text with the
/// values in order of appearance.
pub fn extract(text: &str) -> (String, Vec<f32>) {
    let b = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut values = Vec::new();
    let (mut i, mut copied) = (0, 0);
    while i < b.len() {
        if !(b[i].is_ascii_digit() || b[i] == b'-') {
            i += 1;
            continue;
        }
        let Some(j) = match_end(b, i) else {
            i += 1;
            continue;
        };
        // `i` is an ASCII byte, so it is a char boundary and `b[i - 1]` is the last byte of
        // the previous char: an ASCII letter only if that char is one.
        let glued = i > 0 && b[i - 1].is_ascii_alphabetic();
        if !glued {
            out.push_str(&text[copied..i]);
            out.push_str(NUM_TOKEN);
            // The match is a valid float literal, so parsing cannot fail.
            values.push(text[i..j].parse::<f32>().unwrap_or(0.0));
            copied = j;
        }
        i = j;
    }
    out.push_str(&text[copied..]);
    (out, values)
}

/// Text only (the values are not needed for tokenizer training).
pub fn replace(text: &str) -> String {
    extract(text).0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(s: &str) -> (String, Vec<f32>) {
        extract(s)
    }

    #[test]
    fn basic_cases() {
        assert_eq!(
            ex("alt 12 vy -3 dx 40.5"),
            ("alt [NUM] vy [NUM] dx [NUM]".into(), vec![12.0, -3.0, 40.5])
        );
        assert_eq!(ex("b2 and glm-5"), ("b2 and glm-5".into(), vec![]));
        assert_eq!(ex("3.5km"), ("[NUM]km".into(), vec![3.5]));
        assert_eq!(ex("1,000"), ("[NUM],[NUM]".into(), vec![1.0, 0.0]));
        assert_eq!(ex("1.2.3"), ("[NUM].[NUM]".into(), vec![1.2, 3.0]));
        assert_eq!(ex("7."), ("[NUM].".into(), vec![7.0]));
        assert_eq!(ex("- 3 --4"), ("- [NUM] -[NUM]".into(), vec![3.0, -4.0]));
        assert_eq!(ex(""), (String::new(), vec![]));
    }

    #[test]
    fn negative_zero_keeps_its_sign() {
        let (t, v) = ex("-0");
        assert_eq!(t, "[NUM]");
        assert_eq!(v[0].to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn only_ascii_letters_glue() {
        // A non-ASCII letter before a number does not glue (ASCII rule).
        assert_eq!(ex("é5"), ("é[NUM]".into(), vec![5.0]));
        assert_eq!(ex("x_1"), ("x_[NUM]".into(), vec![1.0]));
        assert_eq!(ex("x1.5y"), ("x1.5y".into(), vec![]));
    }
}
