//! Tensor element types, half-precision conversions and Q8_0 / Q4_0 (de)quantization.

use crate::{Error, Result};

/// Number of elements in one Q8_0 block.
pub const QK8_0: usize = 32;
/// Size in bytes of one Q8_0 block: an f16 scale followed by 32 signed bytes.
pub const Q8_0_BLOCK_BYTES: usize = 2 + QK8_0;
/// Number of elements in one Q4_0 block.
pub const QK4_0: usize = 32;
/// Size in bytes of one Q4_0 block: an f16 scale followed by 16 bytes of 4-bit values.
pub const Q4_0_BLOCK_BYTES: usize = 2 + QK4_0 / 2;

/// ggml tensor types (`enum ggml_type`). F32, F16, BF16, Q8_0, Q4_0 and I8 can be converted to f32;
/// other types are recognized so that files containing them can still be inspected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q8_0,
    BF16,
    /// Raw signed bytes (`GGML_TYPE_I8`), e.g. int8 embeddings or packed ternary weights.
    I8,
    /// Any other ggml type id, kept as is.
    Other(u32),
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            8 => GgmlType::Q8_0,
            24 => GgmlType::I8,
            30 => GgmlType::BF16,
            other => GgmlType::Other(other),
        }
    }

    pub fn to_u32(self) -> u32 {
        match self {
            GgmlType::F32 => 0,
            GgmlType::F16 => 1,
            GgmlType::Q4_0 => 2,
            GgmlType::Q8_0 => 8,
            GgmlType::I8 => 24,
            GgmlType::BF16 => 30,
            GgmlType::Other(v) => v,
        }
    }

    pub fn name(self) -> String {
        match self {
            GgmlType::F32 => "F32".into(),
            GgmlType::F16 => "F16".into(),
            GgmlType::Q4_0 => "Q4_0".into(),
            GgmlType::Q8_0 => "Q8_0".into(),
            GgmlType::BF16 => "BF16".into(),
            GgmlType::I8 => "I8".into(),
            GgmlType::Other(v) => format!("type{v}"),
        }
    }

    /// Elements per block (1 for plain float types).
    pub fn block_len(self) -> Option<usize> {
        match self {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 | GgmlType::I8 => Some(1),
            GgmlType::Q8_0 => Some(QK8_0),
            GgmlType::Q4_0 => Some(QK4_0),
            GgmlType::Other(_) => None,
        }
    }

    /// Bytes per block.
    pub fn block_bytes(self) -> Option<usize> {
        match self {
            GgmlType::F32 => Some(4),
            GgmlType::F16 | GgmlType::BF16 => Some(2),
            GgmlType::I8 => Some(1),
            GgmlType::Q8_0 => Some(Q8_0_BLOCK_BYTES),
            GgmlType::Q4_0 => Some(Q4_0_BLOCK_BYTES),
            GgmlType::Other(_) => None,
        }
    }

    /// Byte size of a tensor of `n_elements` elements of this type.
    pub fn byte_size(self, n_elements: u64) -> Result<u64> {
        let (bl, bb) = match (self.block_len(), self.block_bytes()) {
            (Some(l), Some(b)) => (l as u64, b as u64),
            _ => {
                return Err(Error::Unsupported(format!(
                    "size of ggml type {}",
                    self.name()
                )))
            }
        };
        if !n_elements.is_multiple_of(bl) {
            return Err(Error::Format(format!(
                "{} elements is not a multiple of the {} block size {bl}",
                n_elements,
                self.name()
            )));
        }
        Ok(n_elements / bl * bb)
    }
}

/// IEEE 754 binary16 bits -> f32 (exact).
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // subnormal: normalize
            let mut e = 127 - 15 + 1;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// f32 -> IEEE 754 binary16 bits, round to nearest even (same as numpy / torch).
#[inline]
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0xff {
        // inf / nan
        let nan = if mant != 0 {
            0x200 | (mant >> 13) as u16
        } else {
            0
        };
        return sign | 0x7c00 | nan;
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00; // overflow -> inf
    }
    if e <= 0 {
        // subnormal or zero in f16
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000; // implicit bit
        let shift = (14 - e) as u32; // 1 - e + 13
        let half = 1u32 << (shift - 1);
        let rem = m & ((1u32 << shift) - 1);
        let mut out = m >> shift;
        if rem > half || (rem == half && (out & 1) == 1) {
            out += 1;
        }
        return sign | out as u16;
    }
    let mut out = ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (out & 1) == 1) {
        out += 1; // may carry into the exponent, which is the correct rounding
    }
    sign | out as u16
}

/// bfloat16 bits -> f32 (exact).
#[inline]
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// f32 -> bfloat16 bits, round to nearest even.
#[inline]
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) as u16) | 0x40;
    }
    let lsb = (bits >> 16) & 1;
    ((bits.wrapping_add(0x7fff + lsb)) >> 16) as u16
}

/// Quantize f32 values to ggml Q8_0 blocks (`quantize_row_q8_0_ref`): per block of 32,
/// `d = max|x| / 127`, `q = round(x / d)`, stored as `f16 d` + 32 x `i8`.
/// `x.len()` must be a multiple of 32.
pub fn quantize_q8_0(x: &[f32]) -> Vec<u8> {
    assert!(
        x.len().is_multiple_of(QK8_0),
        "Q8_0 needs a multiple of 32 values, got {}",
        x.len()
    );
    let mut out = vec![0u8; x.len() / QK8_0 * Q8_0_BLOCK_BYTES];
    quantize_q8_0_into(x, &mut out);
    out
}

/// Same as [`quantize_q8_0`] but writes into a caller-provided buffer.
pub fn quantize_q8_0_into(x: &[f32], out: &mut [u8]) {
    assert!(x.len().is_multiple_of(QK8_0));
    assert_eq!(out.len(), x.len() / QK8_0 * Q8_0_BLOCK_BYTES);
    for (blk, dst) in x
        .chunks_exact(QK8_0)
        .zip(out.chunks_exact_mut(Q8_0_BLOCK_BYTES))
    {
        let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        dst[..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        for (q, &v) in dst[2..].iter_mut().zip(blk) {
            *q = ((v * id).round() as i32).clamp(-128, 127) as i8 as u8;
        }
    }
}

/// Quantize f32 values to ggml Q4_0 blocks (`quantize_row_q4_0_ref`): per block of 32, `d` is the
/// value of largest magnitude divided by -8 and `q = min(15, trunc(x / d + 8.5))`, stored as
/// `f16 d` + 16 bytes; byte `j` holds element `j` in its low nibble and element `j + 16` in its
/// high nibble, and the value is `(q - 8) * d`. `x.len()` must be a multiple of 32.
pub fn quantize_q4_0(x: &[f32]) -> Vec<u8> {
    assert!(
        x.len().is_multiple_of(QK4_0),
        "Q4_0 needs a multiple of 32 values, got {}",
        x.len()
    );
    let mut out = vec![0u8; x.len() / QK4_0 * Q4_0_BLOCK_BYTES];
    quantize_q4_0_into(x, &mut out);
    out
}

/// Same as [`quantize_q4_0`] but writes into a caller-provided buffer.
pub fn quantize_q4_0_into(x: &[f32], out: &mut [u8]) {
    assert!(x.len().is_multiple_of(QK4_0));
    assert_eq!(out.len(), x.len() / QK4_0 * Q4_0_BLOCK_BYTES);
    for (blk, dst) in x
        .chunks_exact(QK4_0)
        .zip(out.chunks_exact_mut(Q4_0_BLOCK_BYTES))
    {
        let (d, qs) = q4_0_block(blk);
        dst[..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        dst[2..].copy_from_slice(&qs);
    }
}

/// One Q4_0 block of 32 values: the unrounded scale and the packed values (ggml reference).
pub fn q4_0_block(x: &[f32]) -> (f32, [u8; QK4_0 / 2]) {
    debug_assert_eq!(x.len(), QK4_0);
    // the first value of largest magnitude, with its sign
    let max = x
        .iter()
        .fold(0.0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
    let d = max / -8.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    let q = |v: f32| ((v * id + 8.5) as i8).min(15) as u8;
    let mut qs = [0u8; QK4_0 / 2];
    for (j, b) in qs.iter_mut().enumerate() {
        *b = q(x[j]) | (q(x[j + QK4_0 / 2]) << 4);
    }
    (d, qs)
}

/// Convert raw little-endian tensor bytes of type `ty` to f32.
pub fn dequantize(ty: GgmlType, bytes: &[u8]) -> Result<Vec<f32>> {
    let mut out = Vec::new();
    dequantize_into(ty, bytes, &mut out)?;
    Ok(out)
}

/// Same as [`dequantize`] but appends to `out`.
pub fn dequantize_into(ty: GgmlType, bytes: &[u8], out: &mut Vec<f32>) -> Result<()> {
    let bb = ty
        .block_bytes()
        .ok_or_else(|| Error::Unsupported(format!("dequantize {}", ty.name())))?;
    if !bytes.len().is_multiple_of(bb) {
        return Err(Error::Format(format!(
            "{} bytes is not a multiple of the {} block",
            bytes.len(),
            ty.name()
        )));
    }
    match ty {
        GgmlType::F32 => out.extend(
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        ),
        GgmlType::F16 => out.extend(
            bytes
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))),
        ),
        GgmlType::BF16 => out.extend(
            bytes
                .chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))),
        ),
        GgmlType::Q8_0 => {
            out.reserve(bytes.len() / Q8_0_BLOCK_BYTES * QK8_0);
            for blk in bytes.chunks_exact(Q8_0_BLOCK_BYTES) {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                out.extend(blk[2..].iter().map(|&q| q as i8 as f32 * d));
            }
        }
        GgmlType::Q4_0 => {
            out.reserve(bytes.len() / Q4_0_BLOCK_BYTES * QK4_0);
            for blk in bytes.chunks_exact(Q4_0_BLOCK_BYTES) {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let qs = &blk[2..];
                out.extend(qs.iter().map(|&b| ((b & 0x0f) as i32 - 8) as f32 * d));
                out.extend(qs.iter().map(|&b| ((b >> 4) as i32 - 8) as f32 * d));
            }
        }
        GgmlType::I8 => out.extend(bytes.iter().map(|&b| b as i8 as f32)),
        GgmlType::Other(_) => unreachable!(),
    }
    Ok(())
}

/// Encode f32 values as raw little-endian bytes of type `ty` (F32, F16, BF16, Q8_0, Q4_0, or I8:
/// rounded to the nearest integer and clamped to [-128, 127]).
pub fn encode_f32(ty: GgmlType, x: &[f32]) -> Result<Vec<u8>> {
    Ok(match ty {
        GgmlType::F32 => x.iter().flat_map(|v| v.to_le_bytes()).collect(),
        GgmlType::F16 => x
            .iter()
            .flat_map(|&v| f32_to_f16(v).to_le_bytes())
            .collect(),
        GgmlType::BF16 => x
            .iter()
            .flat_map(|&v| f32_to_bf16(v).to_le_bytes())
            .collect(),
        GgmlType::Q8_0 => {
            if !x.len().is_multiple_of(QK8_0) {
                return Err(Error::Format(format!(
                    "Q8_0 needs a multiple of 32 values, got {}",
                    x.len()
                )));
            }
            quantize_q8_0(x)
        }
        GgmlType::Q4_0 => {
            if !x.len().is_multiple_of(QK4_0) {
                return Err(Error::Format(format!(
                    "Q4_0 needs a multiple of 32 values, got {}",
                    x.len()
                )));
            }
            quantize_q4_0(x)
        }
        GgmlType::I8 => x
            .iter()
            .map(|&v| v.round().clamp(-128.0, 127.0) as i8 as u8)
            .collect(),
        GgmlType::Other(_) => return Err(Error::Unsupported(format!("encode {}", ty.name()))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trip_exact_values() {
        for h in 0u16..=0xffff {
            let f = f16_to_f32(h);
            if f.is_nan() {
                assert!(f16_to_f32(f32_to_f16(f)).is_nan());
                continue;
            }
            assert_eq!(f32_to_f16(f), h, "h={h:#06x} f={f}");
        }
    }

    #[test]
    fn f16_rounding() {
        assert_eq!(f16_to_f32(f32_to_f16(1.0 + 1.0 / 4096.0)), 1.0); // tie -> even
        assert_eq!(f16_to_f32(f32_to_f16(65504.0)), 65504.0);
        assert!(f16_to_f32(f32_to_f16(1e6)).is_infinite());
        assert_eq!(f16_to_f32(f32_to_f16(1e-9)), 0.0);
        let sub = 3.0e-6f32;
        assert!((f16_to_f32(f32_to_f16(sub)) - sub).abs() < 6.0e-8);
    }

    #[test]
    fn i8_round_trip() {
        let v = [0.0f32, 1.0, -1.0, 127.0, -128.0, 300.0, -2.6];
        let b = encode_f32(GgmlType::I8, &v).unwrap();
        assert_eq!(b.len(), v.len());
        assert_eq!(
            dequantize(GgmlType::I8, &b).unwrap(),
            vec![0.0, 1.0, -1.0, 127.0, -128.0, 127.0, -3.0]
        );
        assert_eq!(GgmlType::from_u32(24), GgmlType::I8);
        assert_eq!(GgmlType::I8.byte_size(10).unwrap(), 10);
    }

    /// Five blocks: largest magnitude negative, positive, all zeros, a +3/-3 tie, a fractional scale.
    fn q4_blocks() -> Vec<f32> {
        let mut x = Vec::new();
        x.extend((0..32).map(|i| ((i * 37) % 64 - 31) as f32 / 16.0));
        x.extend((0..32).map(|i| (i - 10) as f32 / 8.0));
        x.extend([0.0; 32]);
        x.extend((0..32).map(|i| match i {
            5 => 3.0,
            9 => -3.0,
            _ => (i % 7 - 3) as f32 * 0.25,
        }));
        x.extend((0..32).map(|i| (((i * 13) % 29 - 14) as f64 * 0.0123) as f32));
        x
    }

    #[test]
    fn q4_0_matches_ggml() {
        // gguf-py `quants.quantize(x, Q4_0)` (the numpy port of `quantize_row_q4_0_ref`).
        let golden = "00b4cf379d047be248bf269df36ad148ae1540b56c5b5b5b4a4a3a39392828281717060600\
                      808888888888888888888888888888888800b69a89797867a0969a897f7867a7969a89832530\
                      a71f96fd74eb52da31b82f960e75fc";
        let want: Vec<u8> = (0..golden.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&golden[i..i + 2], 16).unwrap())
            .collect();
        let x = q4_blocks();
        let q = encode_f32(GgmlType::Q4_0, &x).unwrap();
        assert_eq!(q, want);
        assert_eq!(q.len(), 5 * Q4_0_BLOCK_BYTES);
        assert_eq!(GgmlType::Q4_0.byte_size(64).unwrap(), 36);
        assert_eq!(GgmlType::from_u32(2), GgmlType::Q4_0);
        assert!(encode_f32(GgmlType::Q4_0, &[0.0; 31]).is_err());
    }

    #[test]
    fn q4_0_round_trip() {
        let x = q4_blocks();
        let d = dequantize(GgmlType::Q4_0, &quantize_q4_0(&x)).unwrap();
        assert_eq!(d.len(), x.len());
        for (b, (xb, db)) in x.chunks(32).zip(d.chunks(32)).enumerate() {
            let amax = xb.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let step = amax / 8.0;
            for (i, (&v, &r)) in xb.iter().zip(db).enumerate() {
                // within half a step, or one step on the side clipped at +7 (q ranges over -8..=7),
                // plus the f16 rounding of the scale over 8 steps
                let clipped = v * xb[xb.iter().position(|v| v.abs() == amax).unwrap()] < 0.0;
                let tol = if clipped { step } else { step * 0.5 };
                assert!(
                    (v - r).abs() <= tol + amax * 2e-3,
                    "block {b} value {i}: {v} -> {r}"
                );
            }
            // the value of largest magnitude comes back (nearly) exactly, as -8 * d
            let i = xb.iter().position(|v| v.abs() == amax).unwrap();
            assert!((db[i] - xb[i]).abs() <= amax * 1e-3, "block {b}");
        }
        // exact layout: element j in the low nibble of byte j, element j + 16 in the high one
        let mut v = [0.0f32; 32];
        v[0] = -8.0;
        v[16] = 7.0;
        v[3] = 1.0;
        let q = quantize_q4_0(&v);
        assert_eq!(f16_to_f32(u16::from_le_bytes([q[0], q[1]])), 1.0);
        assert_eq!(q[2], 15 << 4);
        assert_eq!(q[2 + 3], 9 | (8 << 4));
        assert_eq!(dequantize(GgmlType::Q4_0, &q).unwrap(), v.to_vec());
    }

    #[test]
    fn bf16_round_trip() {
        for &v in &[0.0f32, 1.0, -2.5, 3.140625, 1e-30, 1e30] {
            let r = bf16_to_f32(f32_to_bf16(v));
            assert!((r - v).abs() <= v.abs() / 128.0);
        }
        assert_eq!(bf16_to_f32(f32_to_bf16(1.0 + 1.0 / 256.0)), 1.0); // tie -> even
    }
}
