//! Ternary lookup-table products (SPEC-S3 §8, after T-MAC and bitnet.cpp TL kernels): the
//! weights stay at 2 bits each and no weight is ever multiplied.
//!
//! * **Weights.** For every output and every group of 4 inputs, one byte holds two 4-bit masks:
//!   `p` (the inputs whose weight is +1) and `n` (−1); zeros are in neither. A group then
//!   contributes `T[p] − T[n]`, where `T[m] = Σ_{c ∈ m} x_c` is the subset sum of the group's 4
//!   activations. Bytes are laid out in blocks of 16 outputs `[block][group][16]`.
//! * **Activations.** Each row is quantized to int8 with one scale (`s = max|x| / 127`;
//!   optionally a second residual pass `x ≈ s (q + e / 254)`). Its 16-entry tables (`|T| ≤ 508`)
//!   are kept as two byte planes, `T = 256·hi + lo`.
//! * **Products.** For 16 outputs and one group, 4 byte lookups (`lo[p]`, `hi[p]`, `lo[n]`,
//!   `hi[n]`: `vqtbl1q_u8` on NEON, `i8x16.swizzle` on wasm32 SIMD128) and widening int16
//!   accumulation; then `γ_o · s · (256·Σhi + Σlo)` once per output.

// Index loops mirror the SIMD lanes here and read better than iterator chains.
#![allow(clippy::needless_range_loop)]

/// Outputs per block (one 16-byte index vector).
const BLOCK: usize = 16;

/// Inputs per lookup group.
const GROUP: usize = 4;

/// Largest input count: the int16 accumulators of the low bytes hold 128 groups.
pub const MAX_K: usize = 512;

/// A ternary matrix `[n][k]` in lookup-table form (2 bits per weight) with its row scales.
#[derive(Clone, Debug, PartialEq)]
pub struct LutMatrix {
    pub n: usize,
    pub k: usize,
    /// Groups of 4 inputs (`k` padded with zero weights).
    pub groups: usize,
    /// `n.div_ceil(16) * groups * 16` bytes: `p | n << 4` per output and group.
    pub idx: Vec<u8>,
    /// Scale of each output (`n`).
    pub scale: Vec<f32>,
    /// Activation passes: 1 (int8) or 2 (int8 + residual, ~16-bit).
    pub passes: usize,
}

/// Quantized activations and tables of [`LutMatrix::matmul`].
#[derive(Clone, Debug, Default)]
pub struct LutScratch {
    q: Vec<i8>,
    e: Vec<i8>,
    s: Vec<f32>,
    /// `[rows][groups][16]` low and high bytes of the tables.
    lo: Vec<u8>,
    hi: Vec<i8>,
    acc: Vec<i32>,
}

impl LutMatrix {
    /// From trits `t: [n][k]` (each in {−1, 0, +1}) and the row scales. `None` if `k` is larger
    /// than [`MAX_K`].
    pub fn from_trits(t: &[i8], scale: &[f32], n: usize, k: usize, passes: usize) -> Option<Self> {
        if k > MAX_K || t.len() < n * k || scale.len() < n {
            return None;
        }
        let groups = k.div_ceil(GROUP);
        let nb = n.div_ceil(BLOCK);
        let mut idx = vec![0u8; nb * groups * BLOCK];
        for o in 0..n {
            for i in 0..k {
                let (g, c) = (i / GROUP, i % GROUP);
                let b = &mut idx[(o / BLOCK * groups + g) * BLOCK + o % BLOCK];
                match t[o * k + i] {
                    1 => *b |= 1 << c,
                    -1 => *b |= 1 << (4 + c),
                    _ => {}
                }
            }
        }
        Some(LutMatrix {
            n,
            k,
            groups,
            idx,
            scale: scale[..n].to_vec(),
            passes: passes.clamp(1, 2),
        })
    }

    /// Stored bytes (indices and f16 scales).
    pub fn stored_bytes(&self) -> usize {
        self.idx.len() + 2 * self.scale.len()
    }

    /// `y[r][o] (+)= γ_o · Σ_k x[r][k] t[o][k] + bias[o]` for `rows` rows of `x: [rows][k]`.
    pub fn matmul(
        &self,
        x: &[f32],
        rows: usize,
        bias: Option<&[f32]>,
        y: &mut [f32],
        accumulate: bool,
        sc: &mut LutScratch,
    ) {
        let (n, k, g) = (self.n, self.k, self.groups);
        if rows == 0 || n == 0 {
            return;
        }
        assert!(x.len() >= rows * k && y.len() >= rows * n);
        if let Some(b) = bias {
            assert!(b.len() >= n);
        }
        let kg = g * GROUP;
        for (v, len) in [(&mut sc.q, rows * kg), (&mut sc.e, rows * kg)] {
            if v.len() < len {
                v.resize(len, 0);
            }
        }
        if sc.s.len() < rows {
            sc.s.resize(rows, 0.0);
        }
        if sc.lo.len() < rows * g * BLOCK {
            sc.lo.resize(rows * g * BLOCK, 0);
            sc.hi.resize(rows * g * BLOCK, 0);
        }
        let nb = n.div_ceil(BLOCK);
        if sc.acc.len() < rows * nb * BLOCK {
            sc.acc.resize(rows * nb * BLOCK, 0);
        }
        crate::kernels::quantize_rows_i8x2(x, rows, k, kg, &mut sc.q, &mut sc.e, &mut sc.s);
        for pass in 0..self.passes {
            let act = if pass == 0 { &sc.q } else { &sc.e };
            build_tables(act, rows, g, &mut sc.lo, &mut sc.hi);
            let acc = &mut sc.acc[..rows * nb * BLOCK];
            lookups(&self.idx, nb, g, &sc.lo, &sc.hi, rows, acc);
            let f = if pass == 0 { 1.0 } else { 1.0 / 254.0 };
            for r in 0..rows {
                let sr = sc.s[r] * f;
                let yr = &mut y[r * n..(r + 1) * n];
                for o in 0..n {
                    let mut v = acc[r * nb * BLOCK + o] as f32 * sr * self.scale[o];
                    if pass == 0 {
                        v += bias.map_or(0.0, |b| b[o]);
                        if accumulate {
                            v += yr[o];
                        }
                    } else {
                        v += yr[o];
                    }
                    yr[o] = v;
                }
            }
        }
    }
}

/// The 16 subset sums of every group of 4 activations of every row, as low (`u8`) and high
/// (`i8`) byte planes: `T = 256·hi + lo`.
fn build_tables(act: &[i8], rows: usize, groups: usize, lo: &mut [u8], hi: &mut [i8]) {
    for r in 0..rows {
        for g in 0..groups {
            let x = &act[(r * groups + g) * GROUP..(r * groups + g + 1) * GROUP];
            let (x0, x1, x2, x3) = (x[0] as i16, x[1] as i16, x[2] as i16, x[3] as i16);
            let base = (r * groups + g) * BLOCK;
            let mut t = [0i16; 16];
            t[1] = x0;
            t[2] = x1;
            t[3] = x0 + x1;
            for m in 0..4 {
                t[4 + m] = t[m] + x2;
            }
            for m in 0..8 {
                t[8 + m] = t[m] + x3;
            }
            for (m, &v) in t.iter().enumerate() {
                lo[base + m] = v as u8;
                hi[base + m] = (v >> 8) as i8;
            }
        }
    }
}

/// `acc[r][o] = Σ_g T_r,g[p] − T_r,g[n]` for every row and output (blocks of 16 outputs).
#[allow(clippy::too_many_arguments)]
fn lookups(
    idx: &[u8],
    nb: usize,
    groups: usize,
    lo: &[u8],
    hi: &[i8],
    rows: usize,
    acc: &mut [i32],
) {
    assert!(idx.len() >= nb * groups * BLOCK);
    assert!(lo.len() >= rows * groups * BLOCK && hi.len() >= rows * groups * BLOCK);
    assert!(acc.len() >= rows * nb * BLOCK);
    let mut r0 = 0;
    while r0 < rows {
        let rr = (rows - r0).min(4);
        for b in 0..nb {
            // SAFETY: the slices hold the blocks, groups and rows read (asserted above)
            unsafe {
                match rr {
                    4 => lut_tile::<4>(idx, b, groups, lo, hi, r0, nb, acc),
                    3 => lut_tile::<3>(idx, b, groups, lo, hi, r0, nb, acc),
                    2 => lut_tile::<2>(idx, b, groups, lo, hi, r0, nb, acc),
                    _ => lut_tile::<1>(idx, b, groups, lo, hi, r0, nb, acc),
                }
            }
        }
        r0 += rr;
    }
}

/// Rows `r0..r0 + R`, output block `b`, all groups (NEON).
///
/// # Safety
/// `idx`, `lo`, `hi` and `acc` hold the blocks, groups and rows read.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn lut_tile<const R: usize>(
    idx: &[u8],
    b: usize,
    groups: usize,
    lo: &[u8],
    hi: &[i8],
    r0: usize,
    nb: usize,
    acc: &mut [i32],
) {
    use core::arch::aarch64::*;
    let mut alo = [[vdupq_n_s16(0); 2]; R];
    let mut ahi = [[vdupq_n_s16(0); 2]; R];
    let mask = vdupq_n_u8(0x0f);
    for g in 0..groups {
        let iv = vld1q_u8(idx.as_ptr().add((b * groups + g) * BLOCK));
        let (pi, ni) = (vandq_u8(iv, mask), vshrq_n_u8::<4>(iv));
        for r in 0..R {
            let t = ((r0 + r) * groups + g) * BLOCK;
            let tl = vld1q_u8(lo.as_ptr().add(t));
            let th = vld1q_s8(hi.as_ptr().add(t));
            let (lp, ln) = (vqtbl1q_u8(tl, pi), vqtbl1q_u8(tl, ni));
            let (hp, hn) = (vqtbl1q_s8(th, pi), vqtbl1q_s8(th, ni));
            let dl0 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(lp), vget_low_u8(ln)));
            let dl1 = vreinterpretq_s16_u16(vsubl_high_u8(lp, ln));
            alo[r][0] = vaddq_s16(alo[r][0], dl0);
            alo[r][1] = vaddq_s16(alo[r][1], dl1);
            ahi[r][0] = vaddq_s16(ahi[r][0], vsubl_s8(vget_low_s8(hp), vget_low_s8(hn)));
            ahi[r][1] = vaddq_s16(ahi[r][1], vsubl_high_s8(hp, hn));
        }
    }
    for r in 0..R {
        let out = acc.as_mut_ptr().add(((r0 + r) * nb + b) * BLOCK);
        for h in 0..2 {
            let (l, hh) = (alo[r][h], ahi[r][h]);
            let v0 = vaddq_s32(
                vmovl_s16(vget_low_s16(l)),
                vshlq_n_s32::<8>(vmovl_s16(vget_low_s16(hh))),
            );
            let v1 = vaddq_s32(vmovl_high_s16(l), vshlq_n_s32::<8>(vmovl_high_s16(hh)));
            vst1q_s32(out.add(h * 8), v0);
            vst1q_s32(out.add(h * 8 + 4), v1);
        }
    }
}

/// Rows `r0..r0 + R`, output block `b`, all groups (wasm32 SIMD128: `i8x16.swizzle`).
///
/// # Safety
/// `idx`, `lo`, `hi` and `acc` hold the blocks, groups and rows read.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn lut_tile<const R: usize>(
    idx: &[u8],
    b: usize,
    groups: usize,
    lo: &[u8],
    hi: &[i8],
    r0: usize,
    nb: usize,
    acc: &mut [i32],
) {
    use core::arch::wasm32::*;
    let mut alo = [[i16x8_splat(0); 2]; R];
    let mut ahi = [[i16x8_splat(0); 2]; R];
    let mask = u8x16_splat(0x0f);
    for g in 0..groups {
        let iv = v128_load(idx.as_ptr().add((b * groups + g) * BLOCK) as *const v128);
        let (pi, ni) = (v128_and(iv, mask), u8x16_shr(iv, 4));
        for r in 0..R {
            let t = ((r0 + r) * groups + g) * BLOCK;
            let tl = v128_load(lo.as_ptr().add(t) as *const v128);
            let th = v128_load(hi.as_ptr().add(t) as *const v128);
            let (lp, ln) = (i8x16_swizzle(tl, pi), i8x16_swizzle(tl, ni));
            let (hp, hn) = (i8x16_swizzle(th, pi), i8x16_swizzle(th, ni));
            alo[r][0] = i16x8_add(
                alo[r][0],
                i16x8_sub(u16x8_extend_low_u8x16(lp), u16x8_extend_low_u8x16(ln)),
            );
            alo[r][1] = i16x8_add(
                alo[r][1],
                i16x8_sub(u16x8_extend_high_u8x16(lp), u16x8_extend_high_u8x16(ln)),
            );
            ahi[r][0] = i16x8_add(
                ahi[r][0],
                i16x8_sub(i16x8_extend_low_i8x16(hp), i16x8_extend_low_i8x16(hn)),
            );
            ahi[r][1] = i16x8_add(
                ahi[r][1],
                i16x8_sub(i16x8_extend_high_i8x16(hp), i16x8_extend_high_i8x16(hn)),
            );
        }
    }
    for r in 0..R {
        let out = acc.as_mut_ptr().add(((r0 + r) * nb + b) * BLOCK);
        for h in 0..2 {
            let (l, hh) = (alo[r][h], ahi[r][h]);
            let v0 = i32x4_add(
                i32x4_extend_low_i16x8(l),
                i32x4_shl(i32x4_extend_low_i16x8(hh), 8),
            );
            let v1 = i32x4_add(
                i32x4_extend_high_i16x8(l),
                i32x4_shl(i32x4_extend_high_i16x8(hh), 8),
            );
            v128_store(out.add(h * 8) as *mut v128, v0);
            v128_store(out.add(h * 8 + 4) as *mut v128, v1);
        }
    }
}

/// Rows `r0..r0 + R`, output block `b`, all groups (portable).
///
/// # Safety
/// `idx`, `lo`, `hi` and `acc` hold the blocks, groups and rows read.
#[cfg(not(any(
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn lut_tile<const R: usize>(
    idx: &[u8],
    b: usize,
    groups: usize,
    lo: &[u8],
    hi: &[i8],
    r0: usize,
    nb: usize,
    acc: &mut [i32],
) {
    for r in 0..R {
        let out = &mut acc[((r0 + r) * nb + b) * BLOCK..((r0 + r) * nb + b + 1) * BLOCK];
        out.fill(0);
        for g in 0..groups {
            let t = ((r0 + r) * groups + g) * BLOCK;
            for j in 0..BLOCK {
                let i = idx[(b * groups + g) * BLOCK + j];
                let (p, n) = ((i & 15) as usize, (i >> 4) as usize);
                let tp = 256 * hi[t + p] as i32 + lo[t + p] as i32;
                let tn = 256 * hi[t + n] as i32 + lo[t + n] as i32;
                out[j] += tp - tn;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ternary::TernaryMatrix;

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
    fn lookup_products_match_f32() {
        let mut seed = 77;
        for &(rows, k, n) in &[
            (1, 4, 1),
            (3, 7, 5),
            (9, 96, 288),
            (9, 192, 96),
            (6, 128, 384),
            (10, 512, 20),
        ] {
            let x = rnd(rows * k, &mut seed);
            let w = rnd(n * k, &mut seed);
            let b = rnd(n, &mut seed);
            let t = TernaryMatrix::from_f32(&w, n, k);
            let mut yr = vec![0.0; rows * n];
            crate::kernels::scalar::matmul_f32(&x, rows, k, &t.to_f32(), n, Some(&b), &mut yr);
            for (passes, denom) in [(1, 254.0f32), (2, 127.0 * 508.0)] {
                let m = LutMatrix::from_trits(&t.trits(), &t.scale, n, k, passes).unwrap();
                assert!(m.stored_bytes() <= n.div_ceil(16) * 16 * k.div_ceil(4) + 2 * n);
                let mut y = vec![0.0; rows * n];
                let mut sc = LutScratch::default();
                m.matmul(&x, rows, Some(&b), &mut y, false, &mut sc);
                for r in 0..rows {
                    let amax = x[r * k..(r + 1) * k]
                        .iter()
                        .fold(0.0f32, |a, v| a.max(v.abs()));
                    for o in 0..n {
                        let bound = amax / denom * t.scale[o] * k as f32 + 1e-5;
                        let (a, c) = (y[r * n + o], yr[r * n + o]);
                        assert!(
                            (a - c).abs() <= bound,
                            "{passes} {rows}x{k}x{n}: {a} vs {c}"
                        );
                    }
                }
                let y1 = y.clone();
                m.matmul(&x, rows, None, &mut y, true, &mut sc);
                for (i, (a, c)) in y.iter().zip(&y1).enumerate() {
                    assert!((a - (2.0 * c - b[i % n])).abs() < 1e-4);
                }
            }
        }
        assert!(LutMatrix::from_trits(&[0; 513], &[1.0], 1, 513, 1).is_none());
    }
}
