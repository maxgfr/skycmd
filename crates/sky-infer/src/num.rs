//! Native numbers of the `skycmd-s2` models (SPEC-S2 §2): numeric extraction before BPE and the
//! continuous value features `φ(v)` of the `[NUM]` tokens.
//!
//! This mirrors `sky_data::num::extract` bit for bit (same matches, same replacement text, same
//! f32 values) without depending on `sky-data`.

/// Replacement text of an extracted number (a special token of the s2 tokenizers).
pub const NUM_TOKEN: &str = "[NUM]";

/// Number of frequencies `F` of the value features.
pub const NUM_FREQS: usize = 8;

/// Size of `φ(v)`: `1 + 2F` = 17.
pub const NUM_FEATURES: usize = 1 + 2 * NUM_FREQS;

/// Result of [`extract`]: the text with every extracted number replaced by `[NUM]`, and the
/// values in order of appearance.
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
    let start_digits = j;
    while j < b.len() && b[j].is_ascii_digit() {
        j += 1;
    }
    if j == start_digits {
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

/// Numeric extraction (SPEC-S2 §2): every match of `-?\d+(?:\.\d+)?` (ASCII digits, scanned left
/// to right without overlap, like a regex `find_iter`) becomes `[NUM]` and its value is appended
/// to `values`, unless the match is glued to a preceding ASCII letter (`b2`, `glm-5`), in which case
/// the matched text is kept verbatim and scanning resumes after it.
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
        // `i` is an ASCII byte (a char boundary): `b[i - 1]` is an ASCII letter only if the
        // previous char is one
        let glued = i > 0 && b[i - 1].is_ascii_alphabetic();
        if !glued {
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
#[inline]
pub fn omega(k: usize) -> f32 {
    (1u32 << k) as f32 * (std::f32::consts::PI / 8.0)
}

/// Value features `φ(v) = [u, sin(u·ω_0..F-1), cos(u·ω_0..F-1)]` with `u = sign(v)·ln(1+|v|)`
/// (the sines first, then the cosines).
pub fn features(v: f32, out: &mut [f32; NUM_FEATURES]) {
    let u = if v == 0.0 {
        0.0
    } else {
        v.signum() * v.abs().ln_1p()
    };
    out[0] = u;
    // The angles are rounded to f32 like the training graph: since ω_k = 2^k ω_0 exactly,
    // a_k = 2^k a_0 exactly, so sin / cos of a_0 in f64 and 7 double-angle steps (error ~1e-14)
    // give the correctly rounded f32 values of all 8 frequencies for one sin / cos call.
    let a0 = (u * omega(0)) as f64;
    let (mut s, mut c) = a0.sin_cos();
    for k in 0..NUM_FREQS {
        out[1 + k] = s as f32;
        out[1 + NUM_FREQS + k] = c as f32;
        (s, c) = (2.0 * s * c, (c - s) * (c + s));
    }
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
        assert_eq!(ex("x1.5y x_1"), ("x1.5y x_[NUM]".into(), vec![1.0]));
        assert_eq!(extract("-0").values[0].to_bits(), (-0.0f32).to_bits());
        assert_eq!(ex("no digits"), ("no digits".into(), vec![]));
        assert_eq!(ex("1.2.3"), ("[NUM].[NUM]".into(), vec![1.2, 3.0]));
    }

    #[test]
    fn features_shape() {
        let mut f = [0.0; NUM_FEATURES];
        features(0.0, &mut f);
        assert_eq!(f[0], 0.0);
        assert!(f[1..1 + NUM_FREQS].iter().all(|&v| v == 0.0));
        assert!(f[1 + NUM_FREQS..].iter().all(|&v| v == 1.0));
        features(-(std::f32::consts::E - 1.0), &mut f);
        assert!((f[0] + 1.0).abs() < 1e-6);
        assert!((f[1] - (-std::f32::consts::PI / 8.0).sin()).abs() < 1e-6);
    }

    #[test]
    fn double_angle_features_match_direct_sin_cos() {
        let mut f = [0.0; NUM_FEATURES];
        let mut worst = 0.0f32;
        for i in -4000..4000 {
            let v = i as f32 * 0.37 + if i % 3 == 0 { 0.0 } else { 0.125 };
            features(v, &mut f);
            let u = f[0];
            for k in 0..NUM_FREQS {
                let a = (u * omega(k)) as f64;
                worst = worst
                    .max((f[1 + k] - a.sin() as f32).abs())
                    .max((f[1 + NUM_FREQS + k] - a.cos() as f32).abs());
            }
        }
        assert!(worst <= 1.2e-7, "{worst}");
    }
}
