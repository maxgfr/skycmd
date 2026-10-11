//! Native numbers of the skycmd-s2 models (SPEC-S2 §2): numeric extraction
//! before BPE and the continuous value features `φ(v)` of the `[NUM]` tokens.
//!
//! The extraction mirrors the reference `sky_data::num::extract` bit for bit
//! and is tested against its fixtures (`crates/sky-data/tests/fixtures/num_extract.json`);
//! the features are computed in the training graph by [`features`] and on the
//! host by [`features_host`].

use mlx_rs::{error::Exception, ops, Array};

/// Replacement text of an extracted number (a special token of the s2 tokenizers).
pub const NUM_TOKEN: &str = "[NUM]";
/// Fixed id of `[NUM]` in the s2 tokenizers.
pub const NUM_ID: u32 = 5;
/// Number of frequencies `F` of the value features.
pub const NUM_FREQS: usize = 8;
/// Size of `φ(v)`: `1 + 2F` = 17.
pub const NUM_FEATURES: usize = 1 + 2 * NUM_FREQS;

type R<T> = std::result::Result<T, Exception>;

/// Result of [`extract`]: the text with every extracted number replaced by
/// `[NUM]`, and the values in order of appearance.
#[derive(Clone, Debug, PartialEq)]
pub struct Extracted {
    pub text: String,
    pub values: Vec<f32>,
}

/// End of the match of `-?\d+(?:\.\d+)?` starting at byte `i`, if any (ASCII digits).
fn match_at(b: &[u8], i: usize) -> Option<usize> {
    let mut j = i;
    if b.get(j) == Some(&b'-') {
        j += 1;
    }
    let start = j;
    while j < b.len() && b[j].is_ascii_digit() {
        j += 1;
    }
    if j == start {
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

/// Whether the character ending right before byte `i` is an ASCII letter.
fn letter_before(text: &str, i: usize) -> bool {
    i > 0 && text.as_bytes()[i - 1].is_ascii_alphabetic()
}

/// Numeric extraction: every match of `-?\d+(?:\.\d+)?` (scanned left to right
/// without overlap, like a regex `find_iter`) becomes `[NUM]` and its value is
/// appended to `values`, unless the match is glued to a preceding ASCII letter
/// (`b2`, `glm-5`): then the matched text is kept verbatim.
pub fn extract(text: &str) -> Extracted {
    let b = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut values = Vec::new();
    let (mut i, mut copied) = (0, 0);
    while i < b.len() {
        let c = b[i];
        if !(c.is_ascii_digit() || c == b'-') {
            i += 1;
            continue;
        }
        let Some(j) = match_at(b, i) else {
            i += 1;
            continue;
        };
        if !letter_before(text, i) {
            out.push_str(&text[copied..i]);
            out.push_str(NUM_TOKEN);
            values.push(text[i..j].parse::<f32>().unwrap_or(0.0));
            copied = j;
        }
        i = j;
    }
    out.push_str(&text[copied..]);
    Extracted { text: out, values }
}

/// `ω_k = 2^k · π / 8`.
pub fn omega(k: usize) -> f32 {
    (1u32 << k) as f32 * (std::f32::consts::PI / 8.0)
}

/// Signed log `u = sign(v)·ln(1+|v|)`.
pub fn signlog(v: f32) -> f32 {
    if v == 0.0 {
        0.0
    } else {
        v.signum() * v.abs().ln_1p()
    }
}

/// `φ(v) = [u, sin(u·ω_0..F-1), cos(u·ω_0..F-1)]` (sines first, then cosines).
pub fn features_host(v: f32) -> [f32; NUM_FEATURES] {
    let u = signlog(v);
    let mut out = [0f32; NUM_FEATURES];
    out[0] = u;
    for k in 0..NUM_FREQS {
        let a = u * omega(k);
        out[1 + k] = a.sin();
        out[1 + NUM_FREQS + k] = a.cos();
    }
    out
}

/// Signed log of a value array (any shape).
pub fn signlog_array(values: &Array) -> R<Array> {
    ops::sign(values)?.multiply(ops::abs(values)?.log1p()?)
}

/// `φ(v)` of a value array `[..]` -> `[.., 17]` (f32).
pub fn features(values: &Array) -> R<Array> {
    let u = signlog_array(values)?;
    let mut shape = u.shape().to_vec();
    shape.push(1);
    let u = u.reshape(&shape)?;
    let w: Vec<f32> = (0..NUM_FREQS).map(omega).collect();
    let a = u.multiply(Array::from_slice(&w, &[NUM_FREQS as i32]))?;
    ops::concatenate(&[u, ops::sin(&a)?, ops::cos(&a)?], -1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(s: &str) -> (String, Vec<f32>) {
        let e = extract(s);
        (e.text, e.values)
    }

    #[test]
    fn extracts_numbers() {
        assert_eq!(
            ex("alt 12 vy -3 dx 40.5"),
            ("alt [NUM] vy [NUM] dx [NUM]".into(), vec![12.0, -3.0, 40.5])
        );
        assert_eq!(ex("b2 and glm-5"), ("b2 and glm-5".into(), vec![]));
        assert_eq!(
            ex("x=-0.25,7."),
            ("x=[NUM],[NUM].".into(), vec![-0.25, 7.0])
        );
        assert_eq!(ex("1-2"), ("[NUM][NUM]".into(), vec![1.0, -2.0]));
        assert_eq!(ex("- 3 --4"), ("- [NUM] -[NUM]".into(), vec![3.0, -4.0]));
        assert_eq!(ex("é5 (6)"), ("é[NUM] ([NUM])".into(), vec![5.0, 6.0]));
        assert_eq!(ex("level 3: far"), ("level [NUM]: far".into(), vec![3.0]));
        assert_eq!(ex("1.2.3"), ("[NUM].[NUM]".into(), vec![1.2, 3.0]));
        assert_eq!(ex("no digits"), ("no digits".into(), vec![]));
    }

    #[test]
    fn reference_fixtures() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../sky-data/tests/fixtures/num_extract.json");
        let cases: Vec<serde_json::Value> =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert!(cases.len() >= 10);
        for c in cases {
            let e = extract(c["input"].as_str().unwrap());
            assert_eq!(e.text, c["text"].as_str().unwrap(), "{c}");
            let bits: Vec<u32> = e.values.iter().map(|v| v.to_bits()).collect();
            let want: Vec<u32> = c["bits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|b| b.as_u64().unwrap() as u32)
                .collect();
            assert_eq!(bits, want, "{c}");
        }
    }

    #[test]
    fn phi_values() {
        let f = features_host(0.0);
        assert_eq!(f[0], 0.0);
        assert!(f[1..9].iter().all(|&x| x == 0.0));
        assert!(f[9..].iter().all(|&x| x == 1.0));
        // v = e - 1 -> u = 1; v = -(e - 1) -> u = -1.
        let e1 = std::f32::consts::E - 1.0;
        let f = features_host(-e1);
        assert!((f[0] + 1.0).abs() < 1e-6);
        for k in 0..NUM_FREQS {
            let w = (1 << k) as f32 * std::f32::consts::PI / 8.0;
            assert!((f[1 + k] - (-w).sin()).abs() < 1e-5, "sin {k}");
            assert!((f[9 + k] - (-w).cos()).abs() < 1e-5, "cos {k}");
        }
        // Monotone in v through u, and symmetric.
        assert!(features_host(100.0)[0] > features_host(10.0)[0]);
        assert_eq!(features_host(5.0)[0], -features_host(-5.0)[0]);
        // The graph version agrees with the host version.
        let _g = crate::test_util::lock();
        let vals = [0.0f32, 1.0, -3.0, 12.5, 1000.0, -0.25];
        let got = features(&Array::from_slice(&vals, &[2, 3])).unwrap();
        assert_eq!(got.shape(), &[2, 3, 17]);
        got.eval().unwrap();
        let got = got.as_slice::<f32>();
        for (i, &v) in vals.iter().enumerate() {
            let want = features_host(v);
            for j in 0..NUM_FEATURES {
                assert!(
                    (got[i * 17 + j] - want[j]).abs() < 1e-5,
                    "v {v} feature {j}"
                );
            }
        }
    }
}
