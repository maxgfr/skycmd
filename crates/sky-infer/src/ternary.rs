//! Ternary weights of the `skycmd-s2` models (SPEC-S2 §5): every weight is `t · γ_row` with
//! `t ∈ {−1, 0, +1}` and one scale `γ` per output row.
//!
//! **Storage.** A matrix `X` of `rows × cols` is stored as `X.tq`, 2 bits per weight packed 4 per
//! byte (`00` = 0, `01` = +1, `10` = −1; weight `j` of a row in bits `2·(j % 4)` of byte `j / 4`,
//! least significant first), row-major with every row padded to a multiple of 4 weights, plus
//! `X.scale` (one F16 per row).
//!
//! **Compute.** [`TernaryMatrix::matmul`] never multiplies by a weight: each packed byte selects
//! two lane masks from a 256-entry table (sign flip and non-zero), so 4 products become
//! `acc += (x ^ sign) & nonzero` (XOR, AND, ADD on NEON / SIMD128 / SSE2), and the row scale is
//! applied once at the end. [`scalar`] holds the plain reference used by the tests.

// Index loops mirror the SIMD lanes and tiles here and read better than iterator chains.
#![allow(clippy::needless_range_loop)]

use crate::{Error, Result};

/// 2-bit code of a zero weight.
pub const CODE_ZERO: u8 = 0b00;
/// 2-bit code of a +1 weight.
pub const CODE_PLUS: u8 = 0b01;
/// 2-bit code of a −1 weight.
pub const CODE_MINUS: u8 = 0b10;

/// Bytes of one packed row of `cols` weights.
#[inline]
pub fn row_bytes(cols: usize) -> usize {
    cols.div_ceil(4)
}

/// Pack ternary values (`rows × cols`, row-major, each in {−1, 0, +1}) into 2-bit codes.
pub fn pack(trits: &[i8], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(trits.len(), rows * cols, "ternary pack: shape mismatch");
    let rb = row_bytes(cols);
    let mut out = vec![0u8; rows * rb];
    for r in 0..rows {
        for c in 0..cols {
            let code = match trits[r * cols + c] {
                0 => CODE_ZERO,
                t if t > 0 => CODE_PLUS,
                _ => CODE_MINUS,
            };
            out[r * rb + c / 4] |= code << (2 * (c % 4));
        }
    }
    out
}

/// Unpack 2-bit codes into values in {−1, 0, +1}. Fails on the unused code `11`.
pub fn unpack(packed: &[u8], rows: usize, cols: usize) -> Result<Vec<i8>> {
    let rb = row_bytes(cols);
    if packed.len() != rows * rb {
        return Err(Error::Format(format!(
            "ternary matrix {rows}x{cols}: {} bytes, expected {}",
            packed.len(),
            rows * rb
        )));
    }
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            out.push(match (packed[r * rb + c / 4] >> (2 * (c % 4))) & 3 {
                CODE_ZERO => 0,
                CODE_PLUS => 1,
                CODE_MINUS => -1,
                _ => return Err(Error::Format("ternary code 11 is not used".into())),
            });
        }
    }
    Ok(out)
}

/// Absmean ternarization of SPEC-S2 §4: `γ_r = mean(|W_r|)`, `t = clip(round(W / γ_r), −1, 1)`
/// (all zeros when `γ_r = 0`). Returns the values and the per-row scales.
pub fn quantize_absmean(w: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    assert_eq!(w.len(), rows * cols, "ternary quantize: shape mismatch");
    let mut trits = Vec::with_capacity(w.len());
    let mut scales = Vec::with_capacity(rows);
    for row in w.chunks_exact(cols.max(1)).take(rows) {
        let g = row.iter().map(|v| v.abs()).sum::<f32>() / cols.max(1) as f32;
        scales.push(g);
        for &v in row {
            trits.push(if g > 0.0 {
                (v / g).round().clamp(-1.0, 1.0) as i8
            } else {
                0
            });
        }
    }
    (trits, scales)
}

/// Lane masks of one packed byte: `[sign flip (bit 31 set for −1), non-zero (all ones)]`.
const fn lane_masks(b: u8) -> [[u32; 4]; 2] {
    let mut sign = [0u32; 4];
    let mut nz = [0u32; 4];
    let mut j = 0;
    while j < 4 {
        let code = (b >> (2 * j)) & 3;
        if code == CODE_PLUS || code == CODE_MINUS {
            nz[j] = u32::MAX;
        }
        if code == CODE_MINUS {
            sign[j] = 0x8000_0000;
        }
        j += 1;
    }
    [sign, nz]
}

const fn mask_table() -> [[[u32; 4]; 2]; 256] {
    let mut t = [[[0u32; 4]; 2]; 256];
    let mut b = 0;
    while b < 256 {
        t[b] = lane_masks(b as u8);
        b += 1;
    }
    t
}

/// `MASKS[byte] = [sign, nonzero]` lane masks (8 KiB, 16-byte aligned rows).
#[repr(C, align(16))]
struct MaskTable([[[u32; 4]; 2]; 256]);
static MASKS: MaskTable = MaskTable(mask_table());

/// A ternary matrix `[rows][cols]` with its per-row scales.
#[derive(Clone, Debug, PartialEq)]
pub struct TernaryMatrix {
    pub rows: usize,
    pub cols: usize,
    /// `rows * row_bytes(cols)` packed bytes.
    pub packed: Vec<u8>,
    /// One scale per row.
    pub scale: Vec<f32>,
}

impl TernaryMatrix {
    /// From packed codes and row scales (validates the size and the codes).
    pub fn from_packed(packed: Vec<u8>, scale: Vec<f32>, rows: usize, cols: usize) -> Result<Self> {
        if scale.len() != rows {
            return Err(Error::Format(format!(
                "ternary matrix {rows}x{cols}: {} scales",
                scale.len()
            )));
        }
        unpack(&packed, rows, cols)?;
        Ok(TernaryMatrix {
            rows,
            cols,
            packed,
            scale,
        })
    }

    /// Absmean ternarization of an f32 matrix (see [`quantize_absmean`]).
    pub fn from_f32(w: &[f32], rows: usize, cols: usize) -> Self {
        let (t, scale) = quantize_absmean(w, rows, cols);
        TernaryMatrix {
            rows,
            cols,
            packed: pack(&t, rows, cols),
            scale,
        }
    }

    /// Values in {−1, 0, +1}.
    pub fn trits(&self) -> Vec<i8> {
        unpack(&self.packed, self.rows, self.cols).expect("validated at construction")
    }

    /// Dequantized weights `t · γ_row`.
    pub fn to_f32(&self) -> Vec<f32> {
        let t = self.trits();
        let mut out = Vec::with_capacity(t.len());
        for r in 0..self.rows {
            let g = self.scale[r];
            out.extend(
                t[r * self.cols..(r + 1) * self.cols]
                    .iter()
                    .map(|&v| v as f32 * g),
            );
        }
        out
    }

    /// Size of the packed weights and scales in bytes (as stored: scales in f16).
    pub fn stored_bytes(&self) -> usize {
        self.packed.len() + 2 * self.scale.len()
    }

    /// `y[r][o] (+)= γ_o · Σ_k x[r][k] t[o][k] + bias[o]` for `rows` rows of `x: [rows][cols]`
    /// (`y: [rows][self.rows]`), with add/sub only (see the module docs).
    pub fn matmul(
        &self,
        x: &[f32],
        rows: usize,
        bias: Option<&[f32]>,
        y: &mut [f32],
        accumulate: bool,
    ) {
        let (n, k) = (self.rows, self.cols);
        if rows == 0 || n == 0 {
            return;
        }
        assert!(x.len() >= rows * k && y.len() >= rows * n);
        if let Some(b) = bias {
            assert!(b.len() >= n);
        }
        let rb = row_bytes(k);
        let groups = k / 4;
        let tail = k % 4;
        let wrow = |o: usize| &self.packed[o * rb..(o + 1) * rb];
        let tail_sum = |r: usize, o: usize| -> f32 {
            let w = wrow(o);
            let mut s = 0.0;
            for c in groups * 4..k {
                match (w[c / 4] >> (2 * (c % 4))) & 3 {
                    CODE_PLUS => s += x[r * k + c],
                    CODE_MINUS => s -= x[r * k + c],
                    _ => {}
                }
            }
            s
        };
        let mut store = |r: usize, o: usize, s: f32| {
            let v = s * self.scale[o] + bias.map_or(0.0, |b| b[o]);
            let dst = &mut y[r * n + o];
            if accumulate {
                *dst += v;
            } else {
                *dst = v;
            }
        };
        let mut r = 0;
        while r < rows {
            let nr = if r + 4 <= rows { 4 } else { 1 };
            let mut o = 0;
            while o < n {
                let no = if o + 4 <= n { 4 } else { 1 };
                let xp = |i: usize| x[(r + i) * k..].as_ptr();
                let wp = |j: usize| wrow(o + j).as_ptr();
                // SAFETY: rows r..r+nr hold k floats and outputs o..o+no hold `groups` packed
                // bytes (asserted / sliced above)
                match (nr, no) {
                    (4, 4) => {
                        let t = unsafe {
                            tile::<4, 4>(core::array::from_fn(xp), core::array::from_fn(wp), groups)
                        };
                        for i in 0..4 {
                            for j in 0..4 {
                                let s = t[i][j]
                                    + if tail > 0 {
                                        tail_sum(r + i, o + j)
                                    } else {
                                        0.0
                                    };
                                store(r + i, o + j, s);
                            }
                        }
                    }
                    (4, _) => {
                        let t = unsafe { tile::<4, 1>(core::array::from_fn(xp), [wp(0)], groups) };
                        for i in 0..4 {
                            let s = t[i][0] + if tail > 0 { tail_sum(r + i, o) } else { 0.0 };
                            store(r + i, o, s);
                        }
                    }
                    (_, 4) => {
                        let t = unsafe { tile::<1, 4>([xp(0)], core::array::from_fn(wp), groups) };
                        for j in 0..4 {
                            let s = t[0][j] + if tail > 0 { tail_sum(r, o + j) } else { 0.0 };
                            store(r, o + j, s);
                        }
                    }
                    _ => {
                        let t = unsafe { tile::<1, 1>([xp(0)], [wp(0)], groups) };
                        let s = t[0][0] + if tail > 0 { tail_sum(r, o) } else { 0.0 };
                        store(r, o, s);
                    }
                }
                o += no;
            }
            r += nr;
        }
    }
}

/// R x C tile of signed sums over the first `groups` packed bytes (4 weights each).
///
/// # Safety
/// Every `x` pointer must point to `4 * groups` readable floats and every `w` pointer to
/// `groups` readable bytes.
#[inline(always)]
unsafe fn tile<const R: usize, const C: usize>(
    x: [*const f32; R],
    w: [*const u8; C],
    groups: usize,
) -> [[f32; C]; R] {
    let mut acc = [[lanes::zero(); C]; R];
    for g in 0..groups {
        let xv: [lanes::V; R] = core::array::from_fn(|r| lanes::load(x[r].add(4 * g)));
        for c in 0..C {
            let m = &MASKS.0[*w[c].add(g) as usize];
            let (sign, nz) = (lanes::mask(&m[0]), lanes::mask(&m[1]));
            for r in 0..R {
                acc[r][c] = lanes::signed_add(acc[r][c], xv[r], sign, nz);
            }
        }
    }
    core::array::from_fn(|r| core::array::from_fn(|c| lanes::hsum(acc[r][c])))
}

#[cfg(target_arch = "aarch64")]
mod lanes {
    use core::arch::aarch64::*;

    pub type V = float32x4_t;
    pub type M = uint32x4_t;

    #[inline(always)]
    pub fn zero() -> V {
        unsafe { vdupq_n_f32(0.0) }
    }
    /// # Safety
    /// `p` must point to 4 readable f32.
    #[inline(always)]
    pub unsafe fn load(p: *const f32) -> V {
        vld1q_f32(p)
    }
    #[inline(always)]
    pub fn mask(m: &[u32; 4]) -> M {
        unsafe { vld1q_u32(m.as_ptr()) }
    }
    /// `acc + ((x ^ sign) & nz)`
    #[inline(always)]
    pub fn signed_add(acc: V, x: V, sign: M, nz: M) -> V {
        unsafe {
            let t = vandq_u32(veorq_u32(vreinterpretq_u32_f32(x), sign), nz);
            vaddq_f32(acc, vreinterpretq_f32_u32(t))
        }
    }
    #[inline(always)]
    pub fn hsum(v: V) -> f32 {
        unsafe { vaddvq_f32(v) }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod lanes {
    use core::arch::wasm32::*;

    pub type V = v128;
    pub type M = v128;

    #[inline(always)]
    pub fn zero() -> V {
        f32x4_splat(0.0)
    }
    /// # Safety
    /// `p` must point to 4 readable f32.
    #[inline(always)]
    pub unsafe fn load(p: *const f32) -> V {
        v128_load(p as *const v128)
    }
    #[inline(always)]
    pub fn mask(m: &[u32; 4]) -> M {
        unsafe { v128_load(m.as_ptr() as *const v128) }
    }
    /// `acc + ((x ^ sign) & nz)`
    #[inline(always)]
    pub fn signed_add(acc: V, x: V, sign: M, nz: M) -> V {
        f32x4_add(acc, v128_and(v128_xor(x, sign), nz))
    }
    #[inline(always)]
    pub fn hsum(v: V) -> f32 {
        (f32x4_extract_lane::<0>(v) + f32x4_extract_lane::<1>(v))
            + (f32x4_extract_lane::<2>(v) + f32x4_extract_lane::<3>(v))
    }
}

#[cfg(target_arch = "x86_64")]
mod lanes {
    use core::arch::x86_64::*;

    pub type V = __m128;
    pub type M = __m128;

    #[inline(always)]
    pub fn zero() -> V {
        unsafe { _mm_setzero_ps() }
    }
    /// # Safety
    /// `p` must point to 4 readable f32.
    #[inline(always)]
    pub unsafe fn load(p: *const f32) -> V {
        _mm_loadu_ps(p)
    }
    #[inline(always)]
    pub fn mask(m: &[u32; 4]) -> M {
        unsafe { _mm_castsi128_ps(_mm_loadu_si128(m.as_ptr() as *const __m128i)) }
    }
    /// `acc + ((x ^ sign) & nz)`
    #[inline(always)]
    pub fn signed_add(acc: V, x: V, sign: M, nz: M) -> V {
        unsafe { _mm_add_ps(acc, _mm_and_ps(_mm_xor_ps(x, sign), nz)) }
    }
    #[inline(always)]
    pub fn hsum(v: V) -> f32 {
        let mut a = [0f32; 4];
        unsafe { _mm_storeu_ps(a.as_mut_ptr(), v) };
        (a[0] + a[1]) + (a[2] + a[3])
    }
}

#[cfg(not(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
mod lanes {
    pub type V = [f32; 4];
    pub type M = [u32; 4];

    #[inline(always)]
    pub fn zero() -> V {
        [0.0; 4]
    }
    /// # Safety
    /// `p` must point to 4 readable f32.
    #[inline(always)]
    pub unsafe fn load(p: *const f32) -> V {
        [*p, *p.add(1), *p.add(2), *p.add(3)]
    }
    #[inline(always)]
    pub fn mask(m: &[u32; 4]) -> M {
        *m
    }
    /// `acc + ((x ^ sign) & nz)`
    #[inline(always)]
    pub fn signed_add(acc: V, x: V, sign: M, nz: M) -> V {
        core::array::from_fn(|i| acc[i] + f32::from_bits((x[i].to_bits() ^ sign[i]) & nz[i]))
    }
    #[inline(always)]
    pub fn hsum(v: V) -> f32 {
        (v[0] + v[1]) + (v[2] + v[3])
    }
}

/// Plain reference implementation (used by tests to validate the SIMD kernel).
pub mod scalar {
    use super::TernaryMatrix;

    /// Same contract as [`TernaryMatrix::matmul`] (without accumulation), summed in f64.
    pub fn matmul(w: &TernaryMatrix, x: &[f32], rows: usize, bias: Option<&[f32]>, y: &mut [f32]) {
        let t = w.trits();
        let (n, k) = (w.rows, w.cols);
        for r in 0..rows {
            for o in 0..n {
                let mut s = 0.0f64;
                for c in 0..k {
                    s += x[r * k + c] as f64 * t[o * k + c] as f64;
                }
                y[r * n + o] = (s * w.scale[o] as f64) as f32 + bias.map_or(0.0, |b| b[o]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(n: usize, seed: &mut u64) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn pack_round_trip_and_bit_order() {
        let t: Vec<i8> = vec![1, -1, 0, 1, -1, 0, 0];
        let p = pack(&t, 1, 7);
        // weight j in bits 2*(j%4), least significant first; the row is padded to 8 weights
        assert_eq!(p, vec![0b01_00_10_01, 0b00_00_10]);
        assert_eq!(unpack(&p, 1, 7).unwrap(), t);
        assert!(unpack(&[0b11], 1, 1).is_err());
        assert!(unpack(&[0, 0], 1, 1).is_err());
    }

    #[test]
    fn absmean_values() {
        let w = [0.5f32, -0.1, -2.0, 0.0, 0.0, 0.0];
        let (t, s) = quantize_absmean(&w, 2, 3);
        assert_eq!(t, vec![1, 0, -1, 0, 0, 0]);
        assert!((s[0] - 2.6 / 3.0).abs() < 1e-6 && s[1] == 0.0);
        let m = TernaryMatrix::from_f32(&w, 2, 3);
        let d = m.to_f32();
        assert!(d
            .iter()
            .zip(&w)
            .all(|(a, b)| a.abs() < 1e-6 || a.signum() == b.signum()));
    }

    #[test]
    fn simd_matches_scalar() {
        let mut seed = 9;
        for &(rows, k, n) in &[
            (1, 4, 1),
            (3, 7, 5),
            (4, 64, 4),
            (5, 66, 9),
            (9, 128, 13),
            (17, 256, 6),
        ] {
            let x = rnd(rows * k, &mut seed);
            let w = rnd(n * k, &mut seed);
            let b = rnd(n, &mut seed);
            let m = TernaryMatrix::from_f32(&w, n, k);
            let mut y = vec![0.0; rows * n];
            let mut yr = vec![0.0; rows * n];
            m.matmul(&x, rows, Some(&b), &mut y, false);
            scalar::matmul(&m, &x, rows, Some(&b), &mut yr);
            for (a, c) in y.iter().zip(&yr) {
                assert!((a - c).abs() < 1e-4, "{rows}x{k}x{n}: {a} vs {c}");
            }
            // same as the f32 product with the dequantized weights
            let mut yf = vec![0.0; rows * n];
            crate::kernels::scalar::matmul_f32(&x, rows, k, &m.to_f32(), n, Some(&b), &mut yf);
            for (a, c) in y.iter().zip(&yf) {
                assert!((a - c).abs() < 1e-4);
            }
            m.matmul(&x, rows, Some(&b), &mut y, true);
            for (a, c) in y.iter().zip(&yr) {
                assert!((a - 2.0 * c).abs() < 2e-4);
            }
        }
    }
}
