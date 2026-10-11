//! Compute kernels: f32 and Q8_0 matrix products, attention, norms and activations.
//!
//! Layouts are row-major: activations `x[rows][k]`, weights `w[n][k]` (PyTorch `nn.Linear`
//! layout, identical to the GGUF bytes), outputs `y[rows][n]` with `y[r][o] = x[r] . w[o] + b[o]`.
//! Both operands are contiguous along `k`, so every output is a dot product. The products work on
//! 4 x 4 (f32) / 2 x 4 (Q8_0) tiles of rows x outputs with SIMD accumulators: NEON on aarch64
//! (`sdot` when the target has `dotprod`), SIMD128 on wasm32 (`-C target-feature=+simd128`,
//! `i32x4.dot_i16x8`), SSE2 on x86_64, and a portable fallback. Products are cache-blocked so
//! that a panel of weights stays in L1 while the rows stream through.
//! [`scalar`] holds plain reference versions used by the tests.

// Index loops mirror the SIMD lanes and tiles here and read better than iterator chains.
#![allow(clippy::needless_range_loop)]

/// Elements per Q8_0 block.
pub const QK: usize = 32;

/// A Q8_0 block with its scale already converted to f32.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct BlockQ8 {
    pub d: f32,
    pub qs: [i8; QK],
}

/// Quantize one row (or several concatenated rows) of f32 to Q8_0 blocks, like ggml
/// `quantize_row_q8_0_ref` (scale rounded through f16 so it matches GGUF weights bit for bit).
pub fn quantize_q8(x: &[f32], out: &mut [BlockQ8]) {
    debug_assert_eq!(x.len(), out.len() * QK);
    for (blk, dst) in x.chunks_exact(QK).zip(out.iter_mut()) {
        let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        dst.d = sky_gguf::f16_to_f32(sky_gguf::f32_to_f16(d));
        for (q, &v) in dst.qs.iter_mut().zip(blk) {
            *q = (v * id).round().clamp(-128.0, 127.0) as i8;
        }
    }
}

/// How activations are quantized for the Q8_0 products.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ActQuant {
    /// One Q8_0 block per 32 values (ggml `q8_0` activations): fastest.
    Q8,
    /// Q8_0 plus a second Q8_0 pass on the rounding residual at 1/128 of the scale
    /// (`x ~ d * (q + r / 128)`), summed in int32 as `128 * dot(q, w) + dot(r, w)`: ~15-bit
    /// activations, still i8 x i8 -> i32 dot products.
    #[default]
    Q8x2,
}

/// A quantized activation block: scale, values and (for [`ActQuant::Q8x2`]) residual.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct ActBlock {
    pub d: f32,
    pub q: [i8; QK],
    pub r: [i8; QK],
}

/// Quantize one block of activations; channels with their bit set in `mask` are left out.
#[inline]
fn quantize_act_block(src: &[f32], mask: u32, dst: &mut ActBlock, double: bool) {
    let mut tmp = [0.0f32; QK];
    let src: &[f32] = if mask == 0 {
        src
    } else {
        for (i, (t, &v)) in tmp.iter_mut().zip(src).enumerate() {
            if mask & (1 << i) == 0 {
                *t = v;
            }
        }
        &tmp
    };
    let amax = src.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    dst.d = d;
    // ties to even (one instruction on every target, unlike `round`); only exact .5 ties differ
    for i in 0..QK {
        let v = src[i] * id;
        let q = v.round_ties_even();
        dst.q[i] = q as i8;
        if double {
            dst.r[i] = ((v - q) * 128.0).round_ties_even() as i8;
        }
    }
}

/// Quantize `rows` rows of `k` activations (`mask`: per-block excluded channels, or empty).
fn quantize_act(x: &[f32], k: usize, mask: &[u32], out: &mut [ActBlock], double: bool) {
    let nb = k / QK;
    for (row, orow) in x.chunks_exact(k).zip(out.chunks_exact_mut(nb)) {
        for (b, dst) in orow.iter_mut().enumerate() {
            quantize_act_block(
                &row[b * QK..(b + 1) * QK],
                mask.get(b).copied().unwrap_or(0),
                dst,
                double,
            );
        }
    }
}

/// Parse ggml Q8_0 bytes (f16 scale + 32 x i8 per block).
pub fn blocks_from_ggml(bytes: &[u8]) -> Vec<BlockQ8> {
    bytes
        .chunks_exact(sky_gguf::Q8_0_BLOCK_BYTES)
        .map(|b| {
            let mut qs = [0i8; QK];
            for (q, &v) in qs.iter_mut().zip(&b[2..]) {
                *q = v as i8;
            }
            BlockQ8 {
                d: sky_gguf::f16_to_f32(u16::from_le_bytes([b[0], b[1]])),
                qs,
            }
        })
        .collect()
}

/// Serialize blocks to ggml Q8_0 bytes.
pub fn blocks_to_ggml(blocks: &[BlockQ8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(blocks.len() * sky_gguf::Q8_0_BLOCK_BYTES);
    for b in blocks {
        out.extend_from_slice(&sky_gguf::f32_to_f16(b.d).to_le_bytes());
        out.extend(b.qs.iter().map(|&q| q as u8));
    }
    out
}

/// Dequantize blocks to f32.
pub fn dequantize_q8(blocks: &[BlockQ8], out: &mut [f32]) {
    for (b, dst) in blocks.iter().zip(out.chunks_exact_mut(QK)) {
        for (o, &q) in dst.iter_mut().zip(&b.qs) {
            *o = q as f32 * b.d;
        }
    }
}

// ----------------------------------------------------------------------------- SIMD layer

#[cfg(target_arch = "aarch64")]
mod simd {
    use core::arch::aarch64::*;

    pub type V = float32x4_t;
    pub type VI = int32x4_t;

    #[inline(always)]
    pub fn zero() -> V {
        unsafe { vdupq_n_f32(0.0) }
    }
    #[inline(always)]
    pub fn splat(v: f32) -> V {
        unsafe { vdupq_n_f32(v) }
    }
    /// # Safety
    /// `p` must point to 4 readable f32.
    #[inline(always)]
    pub unsafe fn load(p: *const f32) -> V {
        vld1q_f32(p)
    }
    /// # Safety
    /// `p` must point to 4 writable f32.
    #[inline(always)]
    pub unsafe fn store(p: *mut f32, v: V) {
        vst1q_f32(p, v)
    }
    #[inline(always)]
    pub fn fma(acc: V, a: V, b: V) -> V {
        unsafe { vfmaq_f32(acc, a, b) }
    }
    #[inline(always)]
    pub fn hsum(v: V) -> f32 {
        unsafe { vaddvq_f32(v) }
    }
    /// `[hsum(a), hsum(b), hsum(c), hsum(d)]`
    #[inline(always)]
    pub fn hsum4(a: V, b: V, c: V, d: V) -> V {
        unsafe { vpaddq_f32(vpaddq_f32(a, b), vpaddq_f32(c, d)) }
    }

    #[cfg(target_feature = "dotprod")]
    #[inline(always)]
    unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let mut r = acc;
        core::arch::asm!("sdot {0:v}.4s, {1:v}.16b, {2:v}.16b", inout(vreg) r, in(vreg) a, in(vreg) b,
                         options(pure, nomem, nostack));
        r
    }

    /// A 32 x i8 block loaded in registers.
    pub type P = (int8x16_t, int8x16_t);

    #[inline(always)]
    pub fn prep(a: &[i8; 32]) -> P {
        unsafe { (vld1q_s8(a.as_ptr()), vld1q_s8(a.as_ptr().add(16))) }
    }

    /// Partial sums (4 lanes) of the products of two 32 x i8 blocks.
    #[inline(always)]
    pub fn pdot(a: &P, b: &P) -> VI {
        unsafe {
            let (a0, a1) = *a;
            let (b0, b1) = *b;
            #[cfg(target_feature = "dotprod")]
            {
                sdot(sdot(vdupq_n_s32(0), a0, b0), a1, b1)
            }
            #[cfg(not(target_feature = "dotprod"))]
            {
                let p0 = vmull_s8(vget_low_s8(a0), vget_low_s8(b0));
                let p1 = vmull_high_s8(a0, b0);
                let p2 = vmull_s8(vget_low_s8(a1), vget_low_s8(b1));
                let p3 = vmull_high_s8(a1, b1);
                let s = vpaddlq_s16(p0);
                let s = vpadalq_s16(s, p1);
                let s = vpadalq_s16(s, p2);
                vpadalq_s16(s, p3)
            }
        }
    }
    #[inline(always)]
    pub fn i2f(v: VI) -> V {
        unsafe { vcvtq_f32_s32(v) }
    }
    /// `128 * a + b`
    #[inline(always)]
    pub fn shl7_add(a: VI, b: VI) -> VI {
        unsafe { vmlaq_n_s32(b, a, 128) }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd {
    use core::arch::wasm32::*;

    pub type V = v128;
    pub type VI = v128;

    #[inline(always)]
    pub fn zero() -> V {
        f32x4_splat(0.0)
    }
    #[inline(always)]
    pub fn splat(v: f32) -> V {
        f32x4_splat(v)
    }
    /// # Safety
    /// `p` must point to 4 readable f32.
    #[inline(always)]
    pub unsafe fn load(p: *const f32) -> V {
        v128_load(p as *const v128)
    }
    /// # Safety
    /// `p` must point to 4 writable f32.
    #[inline(always)]
    pub unsafe fn store(p: *mut f32, v: V) {
        v128_store(p as *mut v128, v)
    }
    #[inline(always)]
    pub fn fma(acc: V, a: V, b: V) -> V {
        f32x4_add(acc, f32x4_mul(a, b))
    }
    #[inline(always)]
    pub fn hsum(v: V) -> f32 {
        (f32x4_extract_lane::<0>(v) + f32x4_extract_lane::<1>(v))
            + (f32x4_extract_lane::<2>(v) + f32x4_extract_lane::<3>(v))
    }
    /// `[hsum(a), hsum(b), hsum(c), hsum(d)]`
    #[inline(always)]
    pub fn hsum4(a: V, b: V, c: V, d: V) -> V {
        let ab = f32x4_add(
            i32x4_shuffle::<0, 4, 1, 5>(a, b),
            i32x4_shuffle::<2, 6, 3, 7>(a, b),
        );
        let cd = f32x4_add(
            i32x4_shuffle::<0, 4, 1, 5>(c, d),
            i32x4_shuffle::<2, 6, 3, 7>(c, d),
        );
        f32x4_add(
            i32x4_shuffle::<0, 1, 4, 5>(ab, cd),
            i32x4_shuffle::<2, 3, 6, 7>(ab, cd),
        )
    }
    /// A 32 x i8 block widened to 4 x i16x8.
    pub type P = [v128; 4];

    #[inline(always)]
    pub fn prep(a: &[i8; 32]) -> P {
        unsafe {
            let a0 = v128_load(a.as_ptr() as *const v128);
            let a1 = v128_load(a.as_ptr().add(16) as *const v128);
            [
                i16x8_extend_low_i8x16(a0),
                i16x8_extend_high_i8x16(a0),
                i16x8_extend_low_i8x16(a1),
                i16x8_extend_high_i8x16(a1),
            ]
        }
    }

    #[inline(always)]
    pub fn pdot(a: &P, b: &P) -> VI {
        let d0 = i32x4_dot_i16x8(a[0], b[0]);
        let d1 = i32x4_dot_i16x8(a[1], b[1]);
        let d2 = i32x4_dot_i16x8(a[2], b[2]);
        let d3 = i32x4_dot_i16x8(a[3], b[3]);
        i32x4_add(i32x4_add(d0, d1), i32x4_add(d2, d3))
    }
    #[inline(always)]
    pub fn i2f(v: VI) -> V {
        f32x4_convert_i32x4(v)
    }
    /// `128 * a + b`
    #[inline(always)]
    pub fn shl7_add(a: VI, b: VI) -> VI {
        i32x4_add(i32x4_shl(a, 7), b)
    }
}

#[cfg(target_arch = "x86_64")]
mod simd {
    use core::arch::x86_64::*;

    pub type V = __m128;
    pub type VI = __m128i;

    #[inline(always)]
    pub fn zero() -> V {
        unsafe { _mm_setzero_ps() }
    }
    #[inline(always)]
    pub fn splat(v: f32) -> V {
        unsafe { _mm_set1_ps(v) }
    }
    /// # Safety
    /// `p` must point to 4 readable f32.
    #[inline(always)]
    pub unsafe fn load(p: *const f32) -> V {
        _mm_loadu_ps(p)
    }
    /// # Safety
    /// `p` must point to 4 writable f32.
    #[inline(always)]
    pub unsafe fn store(p: *mut f32, v: V) {
        _mm_storeu_ps(p, v)
    }
    #[inline(always)]
    pub fn fma(acc: V, a: V, b: V) -> V {
        unsafe { _mm_add_ps(acc, _mm_mul_ps(a, b)) }
    }
    #[inline(always)]
    pub fn hsum(v: V) -> f32 {
        let mut a = [0f32; 4];
        unsafe { _mm_storeu_ps(a.as_mut_ptr(), v) };
        (a[0] + a[1]) + (a[2] + a[3])
    }
    /// `[hsum(a), hsum(b), hsum(c), hsum(d)]`
    #[inline(always)]
    pub fn hsum4(a: V, b: V, c: V, d: V) -> V {
        unsafe {
            let ab = _mm_add_ps(_mm_unpacklo_ps(a, b), _mm_unpackhi_ps(a, b));
            let cd = _mm_add_ps(_mm_unpacklo_ps(c, d), _mm_unpackhi_ps(c, d));
            _mm_add_ps(_mm_movelh_ps(ab, cd), _mm_movehl_ps(cd, ab))
        }
    }
    /// A 32 x i8 block widened to 4 x i16x8.
    pub type P = [__m128i; 4];

    #[inline(always)]
    pub fn prep(a: &[i8; 32]) -> P {
        unsafe {
            let x0 = _mm_loadu_si128(a.as_ptr() as *const __m128i);
            let x1 = _mm_loadu_si128(a.as_ptr().add(16) as *const __m128i);
            // sign-extend bytes to i16 (SSE2 only): duplicate each byte then shift right
            [
                _mm_srai_epi16(_mm_unpacklo_epi8(x0, x0), 8),
                _mm_srai_epi16(_mm_unpackhi_epi8(x0, x0), 8),
                _mm_srai_epi16(_mm_unpacklo_epi8(x1, x1), 8),
                _mm_srai_epi16(_mm_unpackhi_epi8(x1, x1), 8),
            ]
        }
    }

    #[inline(always)]
    pub fn pdot(a: &P, b: &P) -> VI {
        unsafe {
            let s0 = _mm_add_epi32(_mm_madd_epi16(a[0], b[0]), _mm_madd_epi16(a[1], b[1]));
            let s1 = _mm_add_epi32(_mm_madd_epi16(a[2], b[2]), _mm_madd_epi16(a[3], b[3]));
            _mm_add_epi32(s0, s1)
        }
    }
    #[inline(always)]
    pub fn i2f(v: VI) -> V {
        unsafe { _mm_cvtepi32_ps(v) }
    }
    /// `128 * a + b`
    #[inline(always)]
    pub fn shl7_add(a: VI, b: VI) -> VI {
        unsafe { _mm_add_epi32(_mm_slli_epi32(a, 7), b) }
    }
}

#[cfg(not(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
mod simd {
    pub type V = [f32; 4];
    pub type VI = [i32; 4];

    #[inline(always)]
    pub fn zero() -> V {
        [0.0; 4]
    }
    #[inline(always)]
    pub fn splat(v: f32) -> V {
        [v; 4]
    }
    /// # Safety
    /// `p` must point to 4 readable f32.
    #[inline(always)]
    pub unsafe fn load(p: *const f32) -> V {
        [*p, *p.add(1), *p.add(2), *p.add(3)]
    }
    /// # Safety
    /// `p` must point to 4 writable f32.
    #[inline(always)]
    pub unsafe fn store(p: *mut f32, v: V) {
        for (i, x) in v.iter().enumerate() {
            *p.add(i) = *x;
        }
    }
    #[inline(always)]
    pub fn fma(acc: V, a: V, b: V) -> V {
        [
            acc[0] + a[0] * b[0],
            acc[1] + a[1] * b[1],
            acc[2] + a[2] * b[2],
            acc[3] + a[3] * b[3],
        ]
    }
    #[inline(always)]
    pub fn hsum(v: V) -> f32 {
        (v[0] + v[1]) + (v[2] + v[3])
    }
    /// `[hsum(a), hsum(b), hsum(c), hsum(d)]`
    #[inline(always)]
    pub fn hsum4(a: V, b: V, c: V, d: V) -> V {
        [hsum(a), hsum(b), hsum(c), hsum(d)]
    }
    pub type P = [i8; 32];

    #[inline(always)]
    pub fn prep(a: &[i8; 32]) -> P {
        *a
    }

    #[inline(always)]
    pub fn pdot(a: &P, b: &P) -> VI {
        let mut s = [0i32; 4];
        for i in 0..32 {
            s[i & 3] += a[i] as i32 * b[i] as i32;
        }
        s
    }
    #[inline(always)]
    pub fn i2f(v: VI) -> V {
        [v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32]
    }
    /// `128 * a + b`
    #[inline(always)]
    pub fn shl7_add(a: VI, b: VI) -> VI {
        [
            a[0] * 128 + b[0],
            a[1] * 128 + b[1],
            a[2] * 128 + b[2],
            a[3] * 128 + b[3],
        ]
    }
}

use simd::*;

// ----------------------------------------------------------------------------- f32 products

/// Dot product of two equal-length slices.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let n16 = n / 16 * 16;
    let n4 = n / 4 * 4;
    let (mut s0, mut s1, mut s2, mut s3) = (zero(), zero(), zero(), zero());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut i = 0;
    // SAFETY: all offsets are < n, the length of both slices
    unsafe {
        while i < n16 {
            s0 = fma(s0, load(pa.add(i)), load(pb.add(i)));
            s1 = fma(s1, load(pa.add(i + 4)), load(pb.add(i + 4)));
            s2 = fma(s2, load(pa.add(i + 8)), load(pb.add(i + 8)));
            s3 = fma(s3, load(pa.add(i + 12)), load(pb.add(i + 12)));
            i += 16;
        }
        while i < n4 {
            s0 = fma(s0, load(pa.add(i)), load(pb.add(i)));
            i += 4;
        }
    }
    let mut s = hsum(s0) + hsum(s1) + hsum(s2) + hsum(s3);
    for j in n4..n {
        s += a[j] * b[j];
    }
    s
}

/// `y += a * x`
#[inline]
pub fn axpy(a: f32, x: &[f32], y: &mut [f32]) {
    let n = x.len().min(y.len());
    let n4 = n / 4 * 4;
    let va = splat(a);
    let (px, py) = (x.as_ptr(), y.as_mut_ptr());
    // SAFETY: offsets < n4 <= both lengths
    unsafe {
        let mut i = 0;
        while i < n4 {
            store(py.add(i), fma(load(py.add(i)), va, load(px.add(i))));
            i += 4;
        }
    }
    for j in n4..n {
        y[j] += a * x[j];
    }
}

/// R x C tile of dot products over the first `k4` (multiple of 4) elements.
///
/// # Safety
/// Every pointer of `x` and `w` must point to at least `k4` readable floats.
#[inline(always)]
unsafe fn tile_f32<const R: usize, const C: usize>(
    x: [*const f32; R],
    w: [*const f32; C],
    k4: usize,
) -> [[f32; C]; R] {
    let acc = tile_f32_acc::<R, C>(x, w, k4);
    core::array::from_fn(|r| core::array::from_fn(|c| hsum(acc[r][c])))
}

/// R x C tile of unreduced SIMD accumulators.
///
/// # Safety
/// Every pointer of `x` and `w` must point to at least `k4` readable floats.
#[inline(always)]
unsafe fn tile_f32_acc<const R: usize, const C: usize>(
    x: [*const f32; R],
    w: [*const f32; C],
    k4: usize,
) -> [[V; C]; R] {
    let mut acc = [[zero(); C]; R];
    let mut i = 0;
    while i < k4 {
        // SAFETY: i + 4 <= k4 and the caller guarantees k4 readable floats per pointer
        let xv: [V; R] = core::array::from_fn(|r| load(x[r].add(i)));
        let wv: [V; C] = core::array::from_fn(|c| load(w[c].add(i)));
        for r in 0..R {
            for c in 0..C {
                acc[r][c] = fma(acc[r][c], xv[r], wv[c]);
            }
        }
        i += 4;
    }
    acc
}

/// `y[idx..idx + 4] (+)= v + bias[o..o + 4]`.
#[inline(always)]
fn store4(y: &mut [f32], idx: usize, v: V, bias: Option<&[f32]>, o: usize, accumulate: bool) {
    let dst = &mut y[idx..idx + 4];
    let one = splat(1.0);
    // SAFETY: `dst` and `b[o..o + 4]` are 4 floats (checked by the slicing)
    unsafe {
        let mut v = v;
        if let Some(b) = bias {
            v = fma(v, one, load(b[o..o + 4].as_ptr()));
        }
        if accumulate {
            v = fma(v, one, load(dst.as_ptr()));
        }
        store(dst.as_mut_ptr(), v);
    }
}

/// Bytes of weights / activations kept hot in L1 by the cache blocking of the products.
const L1_BLOCK_BYTES: usize = 32 * 1024;

/// Block sizes (rows, outputs), multiples of 4, so that a block of each operand fits in L1.
#[inline]
fn block_sizes(row_bytes: usize) -> (usize, usize) {
    let b = (L1_BLOCK_BYTES / row_bytes.max(1)).max(4) / 4 * 4;
    (b, b)
}

#[inline(always)]
fn store_out(y: &mut [f32], idx: usize, v: f32, bias: Option<&[f32]>, o: usize, accumulate: bool) {
    let v = v + bias.map_or(0.0, |b| b[o]);
    if accumulate {
        y[idx] += v;
    } else {
        y[idx] = v;
    }
}

/// `y[r][o] (+)= x[r] . w[o] + bias[o]` for `rows` rows; `x: [rows][k]`, `w: [n][k]`, `y: [rows][n]`.
/// With `accumulate` the result is added to `y` (fused residual connection).
#[allow(clippy::too_many_arguments)]
pub fn matmul_f32(
    x: &[f32],
    rows: usize,
    k: usize,
    w: &[f32],
    n: usize,
    bias: Option<&[f32]>,
    y: &mut [f32],
    accumulate: bool,
) {
    matmul_f32_strided(x, k, rows, k, w, k, n, bias, y, n, accumulate);
}

/// [`matmul_f32`] with row strides: row `r` of `x` starts at `r * ldx`, output `o` of `w` at
/// `o * ldw`, and row `r` of `y` at `r * ldy`.
#[allow(clippy::too_many_arguments)]
pub fn matmul_f32_strided(
    x: &[f32],
    ldx: usize,
    rows: usize,
    k: usize,
    w: &[f32],
    ldw: usize,
    n: usize,
    bias: Option<&[f32]>,
    y: &mut [f32],
    ldy: usize,
    accumulate: bool,
) {
    if rows == 0 || n == 0 {
        return;
    }
    assert!(ldx >= k && ldw >= k && ldy >= n);
    assert!(
        x.len() >= (rows - 1) * ldx + k
            && w.len() >= (n - 1) * ldw + k
            && y.len() >= (rows - 1) * ldy + n
    );
    if let Some(b) = bias {
        assert!(b.len() >= n);
    }
    let k4 = k / 4 * 4;
    let tail =
        |r: usize, o: usize| -> f32 { (k4..k).map(|j| x[r * ldx + j] * w[o * ldw + j]).sum() };
    let (rb, ob) = block_sizes(k * 4);
    for r0 in (0..rows).step_by(rb) {
        let r1 = (r0 + rb).min(rows);
        for o0 in (0..n).step_by(ob) {
            let o1 = (o0 + ob).min(n);
            let mut r = r0;
            while r < r1 {
                let nr = if r + 4 <= r1 { 4 } else { 1 };
                let mut o = o0;
                while o < o1 {
                    let no = if o + 4 <= o1 { 4 } else { 1 };
                    // rows r..r+nr and outputs o..o+no have k readable values (asserted above)
                    let xp = |i: usize| x[(r + i) * ldx..].as_ptr();
                    let wp = |j: usize| w[(o + j) * ldw..].as_ptr();
                    match (nr, no) {
                        (4, 4) if k4 == k => {
                            // SAFETY: rows/outputs in range with k readable floats (asserted on entry)
                            let t = unsafe {
                                tile_f32_acc::<4, 4>(
                                    core::array::from_fn(xp),
                                    core::array::from_fn(wp),
                                    k4,
                                )
                            };
                            for (i, t) in t.iter().enumerate() {
                                let v = hsum4(t[0], t[1], t[2], t[3]);
                                store4(y, (r + i) * ldy + o, v, bias, o, accumulate);
                            }
                        }
                        (4, 4) => {
                            // SAFETY: rows/outputs in range with k readable floats (asserted on entry)
                            let t = unsafe {
                                tile_f32::<4, 4>(
                                    core::array::from_fn(xp),
                                    core::array::from_fn(wp),
                                    k4,
                                )
                            };
                            for i in 0..4 {
                                for j in 0..4 {
                                    store_out(
                                        y,
                                        (r + i) * ldy + o + j,
                                        t[i][j] + tail(r + i, o + j),
                                        bias,
                                        o + j,
                                        accumulate,
                                    );
                                }
                            }
                        }
                        (4, _) => {
                            // SAFETY: rows/outputs in range with k readable floats (asserted on entry)
                            let t =
                                unsafe { tile_f32::<4, 1>(core::array::from_fn(xp), [wp(0)], k4) };
                            for i in 0..4 {
                                store_out(
                                    y,
                                    (r + i) * ldy + o,
                                    t[i][0] + tail(r + i, o),
                                    bias,
                                    o,
                                    accumulate,
                                );
                            }
                        }
                        (_, 4) => {
                            // SAFETY: rows/outputs in range with k readable floats (asserted on entry)
                            let t =
                                unsafe { tile_f32::<1, 4>([xp(0)], core::array::from_fn(wp), k4) };
                            for j in 0..4 {
                                store_out(
                                    y,
                                    r * ldy + o + j,
                                    t[0][j] + tail(r, o + j),
                                    bias,
                                    o + j,
                                    accumulate,
                                );
                            }
                        }
                        _ => {
                            let v = dot(&x[r * ldx..r * ldx + k], &w[o * ldw..o * ldw + k]);
                            store_out(y, r * ldy + o, v, bias, o, accumulate);
                        }
                    }
                    o += no;
                }
                r += nr;
            }
        }
    }
}

// ----------------------------------------------------------------------------- Q8_0 products

#[inline(always)]
fn tile_q8<const R: usize, const C: usize, const DOUBLE: bool>(
    x: [&[ActBlock]; R],
    w: [&[BlockQ8]; C],
) -> [[f32; C]; R] {
    let acc = tile_q8_acc::<R, C, DOUBLE>(x, w);
    core::array::from_fn(|r| core::array::from_fn(|c| hsum(acc[r][c])))
}

/// R x C tile of unreduced SIMD accumulators.
#[inline(always)]
fn tile_q8_acc<const R: usize, const C: usize, const DOUBLE: bool>(
    x: [&[ActBlock]; R],
    w: [&[BlockQ8]; C],
) -> [[V; C]; R] {
    let mut acc = [[zero(); C]; R];
    let nb = w[0].len();
    for b in 0..nb {
        // widen the activation blocks once, then stream the outputs one at a time (few live registers)
        let xq: [P; R] = core::array::from_fn(|r| prep(&x[r][b].q));
        let xr: [P; R] = core::array::from_fn(|r| if DOUBLE { prep(&x[r][b].r) } else { xq[r] });
        let xd: [f32; R] = core::array::from_fn(|r| {
            if DOUBLE {
                x[r][b].d * (1.0 / 128.0)
            } else {
                x[r][b].d
            }
        });
        for c in 0..C {
            let wb = &w[c][b];
            let wp = prep(&wb.qs);
            for r in 0..R {
                let s = if DOUBLE {
                    shl7_add(pdot(&xq[r], &wp), pdot(&xr[r], &wp))
                } else {
                    pdot(&xq[r], &wp)
                };
                acc[r][c] = fma(acc[r][c], i2f(s), splat(xd[r] * wb.d));
            }
        }
    }
    acc
}

/// Quantized product: `xq` already quantized (`rows * k/32` blocks), `w: [n][k/32]` blocks.
#[allow(clippy::too_many_arguments)]
fn matmul_q8_impl<const DOUBLE: bool>(
    xq: &[ActBlock],
    rows: usize,
    k: usize,
    w: &[BlockQ8],
    n: usize,
    bias: Option<&[f32]>,
    y: &mut [f32],
    accumulate: bool,
) {
    assert!(k.is_multiple_of(QK));
    let nb = k / QK;
    assert!(xq.len() >= rows * nb && w.len() >= n * nb && y.len() >= rows * n);
    let xrow = |r: usize| &xq[r * nb..r * nb + nb];
    let wrow = |o: usize| &w[o * nb..o * nb + nb];
    let (rb, ob) = block_sizes(nb * core::mem::size_of::<BlockQ8>());
    for r0 in (0..rows).step_by(rb) {
        let r1 = (r0 + rb).min(rows);
        for o0 in (0..n).step_by(ob) {
            let o1 = (o0 + ob).min(n);
            let mut r = r0;
            while r < r1 {
                let nr = if r + 2 <= r1 { 2 } else { 1 };
                let mut o = o0;
                while o < o1 {
                    let no = if o + 4 <= o1 { 4 } else { 1 };
                    match (nr, no) {
                        (2, 4) => {
                            let t = tile_q8_acc::<2, 4, DOUBLE>(
                                [xrow(r), xrow(r + 1)],
                                [wrow(o), wrow(o + 1), wrow(o + 2), wrow(o + 3)],
                            );
                            for (i, t) in t.iter().enumerate() {
                                let v = hsum4(t[0], t[1], t[2], t[3]);
                                store4(y, (r + i) * n + o, v, bias, o, accumulate);
                            }
                        }
                        (2, _) => {
                            let t = tile_q8::<2, 1, DOUBLE>([xrow(r), xrow(r + 1)], [wrow(o)]);
                            store_out(y, r * n + o, t[0][0], bias, o, accumulate);
                            store_out(y, (r + 1) * n + o, t[1][0], bias, o, accumulate);
                        }
                        (_, 4) => {
                            let t = tile_q8::<1, 4, DOUBLE>(
                                [xrow(r)],
                                [wrow(o), wrow(o + 1), wrow(o + 2), wrow(o + 3)],
                            );
                            for j in 0..4 {
                                store_out(y, r * n + o + j, t[0][j], bias, o + j, accumulate);
                            }
                        }
                        _ => {
                            let t = tile_q8::<1, 1, DOUBLE>([xrow(r)], [wrow(o)]);
                            store_out(y, r * n + o, t[0][0], bias, o, accumulate);
                        }
                    }
                    o += no;
                }
                r += nr;
            }
        }
    }
}

/// Input channels of a Q8_0 matrix computed in f32 (LLM.int8()-style decomposition).
#[derive(Clone, Debug, PartialEq)]
pub struct Outliers {
    /// Channel indices.
    pub idx: Vec<usize>,
    /// Weight columns of those channels, `[idx.len()][n_out]`.
    pub cols: Vec<f32>,
    /// Per input block, bit `i` set if channel `32 * b + i` is an outlier.
    pub mask: Vec<u32>,
}

impl Outliers {
    /// Extract the columns `idx` of `w: [n][k]`, zeroing them in `w`.
    pub fn split(w: &mut [f32], n: usize, k: usize, idx: &[usize]) -> Self {
        let mut cols = vec![0.0; idx.len() * n];
        let mut mask = vec![0u32; k.div_ceil(QK)];
        for (j, &c) in idx.iter().enumerate() {
            mask[c / QK] |= 1 << (c % QK);
            for o in 0..n {
                cols[j * n + o] = w[o * k + c];
                w[o * k + c] = 0.0;
            }
        }
        Outliers {
            idx: idx.to_vec(),
            cols,
            mask,
        }
    }

    /// Same as [`Outliers::split`] for a matrix already in Q8_0: the columns are dequantized
    /// into f32 and their quantized values zeroed (the block scales are kept).
    pub fn split_q8(w: &mut [BlockQ8], n: usize, k: usize, idx: &[usize]) -> Self {
        let nb = k / QK;
        let mut cols = vec![0.0; idx.len() * n];
        let mut mask = vec![0u32; nb];
        for (j, &c) in idx.iter().enumerate() {
            let (b, i) = (c / QK, c % QK);
            mask[b] |= 1 << i;
            for o in 0..n {
                let blk = &mut w[o * nb + b];
                cols[j * n + o] = blk.qs[i] as f32 * blk.d;
                blk.qs[i] = 0;
            }
        }
        Outliers {
            idx: idx.to_vec(),
            cols,
            mask,
        }
    }
}

/// `y[r][o] (+)= x[r] . w[o] + bias[o]` with `w` in Q8_0: the activations are quantized into
/// `xq` (`rows * k/32` blocks) with `mode`, then multiplied with i8 x i8 -> i32 dot products.
/// With `outliers = Some((o, start, total))` the channels `o.idx` are left out of the int8
/// product and added in f32 (`w` may be the output rows `start..start + n` of a matrix of
/// `total` rows).
#[allow(clippy::too_many_arguments)]
pub fn matmul_q8(
    x: &[f32],
    rows: usize,
    k: usize,
    w: &[BlockQ8],
    n: usize,
    bias: Option<&[f32]>,
    y: &mut [f32],
    accumulate: bool,
    xq: &mut [ActBlock],
    outliers: Option<(&Outliers, usize, usize)>,
    mode: ActQuant,
) {
    let nb = k / QK;
    let mask: &[u32] = outliers.map_or(&[], |(o, _, _)| &o.mask);
    let double = mode == ActQuant::Q8x2;
    quantize_act(&x[..rows * k], k, mask, &mut xq[..rows * nb], double);
    if double {
        matmul_q8_impl::<true>(xq, rows, k, w, n, bias, y, accumulate);
    } else {
        matmul_q8_impl::<false>(xq, rows, k, w, n, bias, y, accumulate);
    }
    if let Some((outl, out_start, out_total)) = outliers {
        for r in 0..rows {
            let yr = &mut y[r * n..(r + 1) * n];
            for (j, &c) in outl.idx.iter().enumerate() {
                let v = x[r * k + c];
                if v != 0.0 {
                    let col = j * out_total + out_start;
                    axpy(v, &outl.cols[col..col + n], yr);
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------- other ops

/// LayerNorm of each row of `x` (`d` columns) into `out` (may be the same buffer through
/// [`layer_norm_inplace`]).
pub fn layer_norm(x: &[f32], d: usize, w: &[f32], b: Option<&[f32]>, eps: f32, out: &mut [f32]) {
    for (xr, or) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        let (mean, rstd) = moments(xr, eps);
        match b {
            Some(b) => {
                for i in 0..d {
                    or[i] = (xr[i] - mean) * rstd * w[i] + b[i];
                }
            }
            None => {
                for i in 0..d {
                    or[i] = (xr[i] - mean) * rstd * w[i];
                }
            }
        }
    }
}

pub fn layer_norm_inplace(x: &mut [f32], d: usize, w: &[f32], b: Option<&[f32]>, eps: f32) {
    for xr in x.chunks_exact_mut(d) {
        let (mean, rstd) = moments(xr, eps);
        for i in 0..d {
            xr[i] = (xr[i] - mean) * rstd * w[i] + b.map_or(0.0, |b| b[i]);
        }
    }
}

#[inline]
fn moments(x: &[f32], eps: f32) -> (f32, f32) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    (mean, 1.0 / (var + eps).sqrt())
}

/// `e^x` without a libm call (branch-free, vectorizes in loops): Cody-Waite range reduction
/// and a degree-6 polynomial, relative error < 2e-7 on the whole f32 range.
#[inline(always)]
pub fn fast_exp(x: f32) -> f32 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.693_145_75;
    const LN2_LO: f32 = 1.428_606_8e-6;
    const ROUND: f32 = 12_582_912.0; // 1.5 * 2^23: adding it rounds to an integer
    let x = x.clamp(-87.0, 88.0);
    let n = (x * LOG2E + ROUND) - ROUND;
    let r = (x - n * LN2_HI) - n * LN2_LO;
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (1.0 / 6.0 + r * (1.0 / 24.0 + r * (1.0 / 120.0 + r * (1.0 / 720.0))))));
    p * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// erf(x), Abramowitz & Stegun 7.1.26 (max abs error ~1.5e-7).
#[inline]
pub fn erf(x: f32) -> f32 {
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * a);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_74) * t
            + 0.254_829_6)
            * t
            * fast_exp(-a * a);
    s * y
}

/// Exact (erf) GELU, as `torch.nn.GELU()` and HF `gelu`.
#[inline]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// Softmax in place.
pub fn softmax(x: &mut [f32]) {
    let m = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut s = 0.0;
    for v in x.iter_mut() {
        *v = fast_exp(*v - m);
        s += *v;
    }
    let inv = 1.0 / s;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Precomputed RoPE tables (`cos`, `sin` of shape `[max_pos][head_dim/2]`), computed like HF
/// (`inv_freq = 1 / theta^(2i/d)` and the angles in f32).
#[derive(Clone, Debug)]
pub struct Rope {
    pub half: usize,
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
}

impl Rope {
    pub fn new(head_dim: usize, theta: f32, max_pos: usize) -> Self {
        let half = head_dim / 2;
        let inv: Vec<f32> = (0..half)
            .map(|i| 1.0 / theta.powf((2 * i) as f32 / head_dim as f32))
            .collect();
        let mut cos = Vec::with_capacity(max_pos * half);
        let mut sin = Vec::with_capacity(max_pos * half);
        for p in 0..max_pos {
            for &f in &inv {
                let a = p as f32 * f;
                cos.push((a as f64).cos() as f32);
                sin.push((a as f64).sin() as f32);
            }
        }
        Rope { half, cos, sin }
    }

    /// Rotate one head vector at position `pos` (rotate_half / NeoX style).
    #[inline]
    pub fn apply(&self, v: &mut [f32], pos: usize) {
        let h = self.half;
        let c = &self.cos[pos * h..pos * h + h];
        let s = &self.sin[pos * h..pos * h + h];
        let (a, b) = v.split_at_mut(h);
        for i in 0..h {
            let (x1, x2) = (a[i], b[i]);
            a[i] = x1 * c[i] - x2 * s[i];
            b[i] = x2 * c[i] + x1 * s[i];
        }
    }
}

/// Scratch buffers of [`Attention::run`].
#[derive(Clone, Debug, Default)]
pub struct AttnScratch {
    scores: Vec<f32>,
    vt: Vec<f32>,
}

/// Queries per block in [`Attention::run`].
const ATTN_QBLOCK: usize = 16;

impl AttnScratch {
    pub fn new(max_kv: usize, head_dim: usize) -> Self {
        AttnScratch {
            scores: vec![0.0; ATTN_QBLOCK * max_kv],
            vt: vec![0.0; head_dim * max_kv],
        }
    }
}

/// Bidirectional scaled dot-product attention (all heads) over `n_kv` key/value rows.
pub struct Attention<'a> {
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub kv_stride: usize,
    pub n_kv: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    /// Symmetric window: keys with `|i - j| <= window` only.
    pub window: Option<usize>,
}

impl<'a> Attention<'a> {
    /// Attend the query rows `q[i * q_stride..]` (positions `pos[i]`, increasing) and write
    /// `out[i * out_stride..]`. Per head and per block of queries, the scores are one product
    /// restricted to the keys of the block's windows, then one product with the transposed values.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        q: &[f32],
        q_stride: usize,
        pos: &[usize],
        out: &mut [f32],
        out_stride: usize,
        sc: &mut AttnScratch,
    ) {
        let (hd, n_kv) = (self.head_dim, self.n_kv);
        let scale = 1.0 / (hd as f32).sqrt();
        for h in 0..self.n_heads {
            // V of this head, transposed: vt[c][j]
            let vt = &mut sc.vt[..hd * n_kv];
            for j in 0..n_kv {
                let src = &self.v[j * self.kv_stride + h * hd..j * self.kv_stride + (h + 1) * hd];
                for (c, &x) in src.iter().enumerate() {
                    vt[c * n_kv + j] = x;
                }
            }
            for (b, qpos) in pos.chunks(ATTN_QBLOCK).enumerate() {
                let i0 = b * ATTN_QBLOCK;
                let nq = qpos.len();
                let (lo, hi) = match self.window {
                    Some(w) => (qpos[0].saturating_sub(w), (qpos[nq - 1] + w + 1).min(n_kv)),
                    None => (0, n_kv),
                };
                let len = hi - lo;
                let s = &mut sc.scores[..nq * len];
                matmul_f32_strided(
                    &q[i0 * q_stride + h * hd..],
                    q_stride,
                    nq,
                    hd,
                    &self.k[lo * self.kv_stride + h * hd..],
                    self.kv_stride,
                    len,
                    None,
                    s,
                    len,
                    false,
                );
                for (r, &p) in qpos.iter().enumerate() {
                    let row = &mut s[r * len..(r + 1) * len];
                    for (j, v) in row.iter_mut().enumerate() {
                        let kj = lo + j;
                        let inside = self.window.is_none_or(|w| kj + w >= p && kj <= p + w);
                        *v = if inside {
                            *v * scale
                        } else {
                            f32::NEG_INFINITY
                        };
                    }
                    softmax(row);
                }
                matmul_f32_strided(
                    s,
                    len,
                    nq,
                    len,
                    &vt[lo..],
                    n_kv,
                    hd,
                    None,
                    &mut out[i0 * out_stride + h * hd..],
                    out_stride,
                    false,
                );
            }
        }
    }
}

/// Plain reference implementations (used by tests to validate the SIMD kernels).
pub mod scalar {
    use super::{ActBlock, BlockQ8};

    pub fn matmul_f32(
        x: &[f32],
        rows: usize,
        k: usize,
        w: &[f32],
        n: usize,
        bias: Option<&[f32]>,
        y: &mut [f32],
    ) {
        for r in 0..rows {
            for o in 0..n {
                let mut s = 0.0f64;
                for i in 0..k {
                    s += x[r * k + i] as f64 * w[o * k + i] as f64;
                }
                y[r * n + o] = s as f32 + bias.map_or(0.0, |b| b[o]);
            }
        }
    }

    /// Reference for [`super::matmul_q8`] on already quantized activations.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_q8(
        xq: &[ActBlock],
        rows: usize,
        k: usize,
        w: &[BlockQ8],
        n: usize,
        bias: Option<&[f32]>,
        y: &mut [f32],
        double: bool,
    ) {
        let nb = k / super::QK;
        for r in 0..rows {
            for o in 0..n {
                let mut s = 0.0f64;
                for b in 0..nb {
                    let (xb, wb) = (&xq[r * nb + b], &w[o * nb + b]);
                    let iq: i64 =
                        xb.q.iter()
                            .zip(&wb.qs)
                            .map(|(&a, &c)| a as i64 * c as i64)
                            .sum();
                    let ir: i64 =
                        xb.r.iter()
                            .zip(&wb.qs)
                            .map(|(&a, &c)| a as i64 * c as i64)
                            .sum();
                    let v = if double {
                        iq as f64 + ir as f64 / 128.0
                    } else {
                        iq as f64
                    };
                    s += v * xb.d as f64 * wb.d as f64;
                }
                y[r * n + o] = s as f32 + bias.map_or(0.0, |b| b[o]);
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
    fn simd_matmul_matches_scalar() {
        let mut seed = 7;
        for &(rows, k, n) in &[
            (1, 4, 1),
            (3, 7, 5),
            (4, 64, 4),
            (9, 96, 13),
            (17, 256, 33),
            (6, 36, 2),
        ] {
            let x = rnd(rows * k, &mut seed);
            let w = rnd(n * k, &mut seed);
            let b = rnd(n, &mut seed);
            let mut y = vec![0.0; rows * n];
            let mut yr = vec![0.0; rows * n];
            matmul_f32(&x, rows, k, &w, n, Some(&b), &mut y, false);
            scalar::matmul_f32(&x, rows, k, &w, n, Some(&b), &mut yr);
            for (a, c) in y.iter().zip(&yr) {
                assert!((a - c).abs() < 1e-4, "{rows}x{k}x{n}: {a} vs {c}");
            }
            // accumulate adds on top
            matmul_f32(&x, rows, k, &w, n, Some(&b), &mut y, true);
            for (a, c) in y.iter().zip(&yr) {
                assert!((a - 2.0 * c).abs() < 2e-4);
            }
        }
    }

    #[test]
    fn simd_q8_matches_scalar() {
        let mut seed = 11;
        for &(rows, k, n) in &[(1, 32, 1), (2, 64, 4), (5, 96, 7), (8, 256, 12)] {
            let x = rnd(rows * k, &mut seed);
            let w = rnd(n * k, &mut seed);
            let mut wq = vec![BlockQ8::default(); n * k / QK];
            quantize_q8(&w, &mut wq);
            let mut yf = vec![0.0; rows * n];
            scalar::matmul_f32(&x, rows, k, &w, n, None, &mut yf);
            // same weights, exact activations: isolates the activation rounding error
            let mut wd = vec![0.0; n * k];
            dequantize_q8(&wq, &mut wd);
            let mut yw = vec![0.0; rows * n];
            scalar::matmul_f32(&x, rows, k, &wd, n, None, &mut yw);
            let mut errs = Vec::new();
            for mode in [ActQuant::Q8, ActQuant::Q8x2] {
                let mut xq = vec![ActBlock::default(); rows * k / QK];
                let mut y = vec![0.0; rows * n];
                matmul_q8(
                    &x, rows, k, &wq, n, None, &mut y, false, &mut xq, None, mode,
                );
                let mut yr = vec![0.0; rows * n];
                scalar::matmul_q8(&xq, rows, k, &wq, n, None, &mut yr, mode == ActQuant::Q8x2);
                let mut e = 0.0f32;
                for i in 0..y.len() {
                    assert!(
                        (y[i] - yr[i]).abs() < 1e-4,
                        "{mode:?}: {} vs {}",
                        y[i],
                        yr[i]
                    );
                    assert!(
                        (y[i] - yf[i]).abs() < 0.02 * (k as f32).sqrt(),
                        "q8 too far from f32"
                    );
                    e = e.max((y[i] - yw[i]).abs());
                }
                errs.push(e);
            }
            // the residual pass removes most of the activation rounding error
            assert!(errs[1] < 0.05 * errs[0], "{errs:?}");
        }
    }

    #[test]
    fn outlier_decomposition_is_exact_for_outliers() {
        let mut seed = 21;
        let (rows, k, n) = (3, 64, 8);
        let mut x = rnd(rows * k, &mut seed);
        for r in 0..rows {
            x[r * k + 9] = 25.0;
        }
        let w = rnd(n * k, &mut seed);
        let mut wz = w.clone();
        let outl = Outliers::split(&mut wz, n, k, &[9]);
        let mut wq = vec![BlockQ8::default(); n * k / QK];
        quantize_q8(&wz, &mut wq);
        let mut xq = vec![ActBlock::default(); rows * k / QK];
        let mut y = vec![0.0; rows * n];
        matmul_q8(
            &x,
            rows,
            k,
            &wq,
            n,
            None,
            &mut y,
            false,
            &mut xq,
            Some((&outl, 0, n)),
            ActQuant::Q8,
        );
        let mut plain = vec![0.0; rows * n];
        let mut wq2 = vec![BlockQ8::default(); n * k / QK];
        quantize_q8(&w, &mut wq2);
        matmul_q8(
            &x,
            rows,
            k,
            &wq2,
            n,
            None,
            &mut plain,
            false,
            &mut xq,
            None,
            ActQuant::Q8,
        );
        let mut yf = vec![0.0; rows * n];
        scalar::matmul_f32(&x, rows, k, &w, n, None, &mut yf);
        let err = |a: &[f32]| {
            a.iter()
                .zip(&yf)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0, f32::max)
        };
        assert!(
            err(&y) < 0.5 * err(&plain),
            "outliers {} vs plain {}",
            err(&y),
            err(&plain)
        );
    }

    #[test]
    fn q8_ggml_round_trip() {
        let mut seed = 3;
        let x = rnd(64, &mut seed);
        let mut b = vec![BlockQ8::default(); 2];
        quantize_q8(&x, &mut b);
        assert_eq!(blocks_from_ggml(&blocks_to_ggml(&b)), b);
        assert_eq!(blocks_to_ggml(&b), sky_gguf::quantize_q8_0(&x));
    }

    #[test]
    fn fast_exp_accuracy() {
        let mut x = -87.0f32;
        while x < 88.0 {
            let (a, b) = (fast_exp(x), (x as f64).exp());
            assert!(((a as f64 - b) / b).abs() < 3e-7, "exp({x}) = {a} vs {b}");
            x += 0.173;
        }
        assert_eq!(fast_exp(0.0), 1.0);
        assert!(fast_exp(f32::NEG_INFINITY) < 1e-37);
    }

    #[test]
    fn attention_matches_naive() {
        let mut seed = 5;
        let (t, nh, hd) = (37, 2, 8);
        let d = nh * hd;
        let qkv = rnd(t * 3 * d, &mut seed);
        for window in [None, Some(4)] {
            let a = Attention {
                k: &qkv[d..],
                v: &qkv[2 * d..],
                kv_stride: 3 * d,
                n_kv: t,
                n_heads: nh,
                head_dim: hd,
                window,
            };
            let pos: Vec<usize> = (0..t).collect();
            let mut out = vec![0.0; t * d];
            let mut sc = AttnScratch::new(t, hd);
            a.run(&qkv, 3 * d, &pos, &mut out, d, &mut sc);
            for i in 0..t {
                for h in 0..nh {
                    let js: Vec<usize> = (0..t)
                        .filter(|&j| window.is_none_or(|w| i.abs_diff(j) <= w))
                        .collect();
                    let mut sv: Vec<f32> = js
                        .iter()
                        .map(|&j| {
                            (0..hd)
                                .map(|c| {
                                    qkv[i * 3 * d + h * hd + c] * qkv[j * 3 * d + d + h * hd + c]
                                })
                                .sum::<f32>()
                                / (hd as f32).sqrt()
                        })
                        .collect();
                    softmax(&mut sv);
                    for c in 0..hd {
                        let want: f32 = js
                            .iter()
                            .zip(&sv)
                            .map(|(&j, &p)| p * qkv[j * 3 * d + 2 * d + h * hd + c])
                            .sum();
                        assert!(
                            (out[i * d + h * hd + c] - want).abs() < 1e-5,
                            "{window:?} {i} {h} {c}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn erf_and_gelu() {
        for &(x, e) in &[
            (0.0f32, 0.0f32),
            (0.5, 0.520_499_9),
            (1.0, 0.842_700_8),
            (-2.0, -0.995_322_3),
            (3.5, 0.999_999_3),
        ] {
            assert!((erf(x) - e).abs() < 3e-7, "erf({x}) = {}", erf(x));
        }
        assert!((gelu(1.0) - 0.841_344_7).abs() < 1e-6);
    }
}
