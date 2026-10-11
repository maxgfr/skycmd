//! Tensor element types, half-precision conversions and Q8_0 (de)quantization.

use crate::{Error, Result};

/// Number of elements in one Q8_0 block.
pub const QK8_0: usize = 32;
/// Size in bytes of one Q8_0 block: an f16 scale followed by 32 signed bytes.
pub const Q8_0_BLOCK_BYTES: usize = 2 + QK8_0;

/// ggml tensor types (`enum ggml_type`). Only F32, F16, BF16 and Q8_0 can be converted
/// to f32; other types are recognized so that files containing them can still be inspected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32,
    F16,
    Q8_0,
    BF16,
    /// Any other ggml type id, kept as is.
    Other(u32),
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            8 => GgmlType::Q8_0,
            30 => GgmlType::BF16,
            other => GgmlType::Other(other),
        }
    }

    pub fn to_u32(self) -> u32 {
        match self {
            GgmlType::F32 => 0,
            GgmlType::F16 => 1,
            GgmlType::Q8_0 => 8,
            GgmlType::BF16 => 30,
            GgmlType::Other(v) => v,
        }
    }

    pub fn name(self) -> String {
        match self {
            GgmlType::F32 => "F32".into(),
            GgmlType::F16 => "F16".into(),
            GgmlType::Q8_0 => "Q8_0".into(),
            GgmlType::BF16 => "BF16".into(),
            GgmlType::Other(v) => format!("type{v}"),
        }
    }

    /// Elements per block (1 for plain float types).
    pub fn block_len(self) -> Option<usize> {
        match self {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => Some(1),
            GgmlType::Q8_0 => Some(QK8_0),
            GgmlType::Other(_) => None,
        }
    }

    /// Bytes per block.
    pub fn block_bytes(self) -> Option<usize> {
        match self {
            GgmlType::F32 => Some(4),
            GgmlType::F16 | GgmlType::BF16 => Some(2),
            GgmlType::Q8_0 => Some(Q8_0_BLOCK_BYTES),
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
        GgmlType::Other(_) => unreachable!(),
    }
    Ok(())
}

/// Encode f32 values as raw little-endian bytes of type `ty` (F32, F16, BF16 or Q8_0).
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
    fn bf16_round_trip() {
        for &v in &[0.0f32, 1.0, -2.5, 3.140625, 1e-30, 1e30] {
            let r = bf16_to_f32(f32_to_bf16(v));
            assert!((r - v).abs() <= v.abs() / 128.0);
        }
        assert_eq!(bf16_to_f32(f32_to_bf16(1.0 + 1.0 / 256.0)), 1.0); // tie -> even
    }
}
