//! Expand-and-sparsify feed-forward (SPEC-S3 §9 ablation; GGUF keys `skycmd-s2.ffn_es_seed`,
//! `ffn_es_units`, `ffn_es_k`, `ffn_es_fan_in`).
//!
//! An encoder block's FFN becomes `x + W_down · kWTA(P · LN(x))`, where `P` (`units × d`) is a
//! fixed sparse ±1 matrix generated from the u64 seed (no weights in the file): unit `j` reads
//! `fan_in` distinct inputs with weights `±1/√fan_in`. k-WTA keeps each token's units `≥` its k-th
//! largest pre-activation. The forward pass computes the pre-activations of every row with one
//! panel product (the fixed matrix is only 12.5 % non-zero at fan_in 8 of 64, but a dense SIMD
//! product beats scalar gathers by far at these sizes), finds the k-th largest value with a
//! k-element running minimum, and reads only the active columns of `W_down` (stored transposed,
//! one contiguous row per unit): `k · d` multiply-adds instead of `d · units`.
//!
//! The generator is the one of `sky-train` (`s2::es`), bit for bit.

use crate::kernels::PanelMatrix;

/// splitmix64 step.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The fixed projection: for each of the `units` rows, `fan_in` pairs `(input index, weight)`
/// with weight `±1/√fan_in`; the input indices of a row are distinct, in draw order.
pub fn projection(seed: u64, d: usize, units: usize, fan_in: usize) -> Vec<(usize, f32)> {
    assert!(fan_in >= 1 && fan_in <= d, "fan_in must be in 1..=d");
    let scale = 1.0 / (fan_in as f32).sqrt();
    let mut state = seed;
    let mut out = Vec::with_capacity(units * fan_in);
    for _ in 0..units {
        let start = out.len();
        while out.len() - start < fan_in {
            let r = splitmix64(&mut state);
            let i = ((r >> 32) % d as u64) as usize;
            if out[start..].iter().any(|&(j, _)| j == i) {
                continue;
            }
            out.push((i, if r & 1 == 1 { -scale } else { scale }));
        }
    }
    out
}

/// Hyper-parameters of the expand-and-sparsify FFN as stored in the GGUF.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EsSpec {
    pub seed: u64,
    /// Expansion units (`ffn_down` has this many inputs).
    pub units: usize,
    /// Active units per token.
    pub k: usize,
    /// Inputs per unit.
    pub fan_in: usize,
}

/// The expand-and-sparsify FFN of one block, ready to run.
#[derive(Clone, Debug)]
pub struct EsFfn {
    pub units: usize,
    pub k: usize,
    pub fan_in: usize,
    /// The fixed projection as a dense panel matrix (`units` outputs, `d` inputs).
    up: PanelMatrix,
    /// `W_downᵀ`, `[units][d]`.
    down_t: Vec<f32>,
    d: usize,
}

impl EsFfn {
    /// `down` is `ffn_down` as row-major f32 `[d][units]`.
    pub fn new(spec: EsSpec, d: usize, down: &[f32]) -> Self {
        assert_eq!(down.len(), d * spec.units);
        let mut dense = vec![0f32; spec.units * d];
        for (n, &(i, w)) in projection(spec.seed, d, spec.units, spec.fan_in)
            .iter()
            .enumerate()
        {
            dense[(n / spec.fan_in) * d + i] = w;
        }
        let mut down_t = vec![0f32; spec.units * d];
        for o in 0..d {
            for j in 0..spec.units {
                down_t[j * d + o] = down[o * spec.units + j];
            }
        }
        EsFfn {
            units: spec.units,
            k: spec.k,
            fan_in: spec.fan_in,
            up: PanelMatrix::from_rows(&dense, spec.units, d),
            down_t,
            d,
        }
    }

    /// `x[r] += W_down · kWTA(P · h[r])` for `rows` rows; `u` is scratch of at least
    /// `rows · units` floats and `cand` of at least `units`.
    pub fn forward(&self, h: &[f32], rows: usize, x: &mut [f32], u: &mut [f32], cand: &mut [f32]) {
        let (d, n, k) = (self.d, self.units, self.k);
        let u = &mut u[..rows * n];
        self.up.matmul(h, d, rows, None, u, n, false);
        for r in 0..rows {
            let ur = &u[r * n..(r + 1) * n];
            let thr = kth_largest(ur, k, cand);
            let xr = &mut x[r * d..(r + 1) * d];
            for (j, &uj) in ur.iter().enumerate() {
                if uj >= thr {
                    let col = &self.down_t[j * d..(j + 1) * d];
                    for (o, &c) in xr.iter_mut().zip(col) {
                        *o += uj * c;
                    }
                }
            }
        }
    }
}

/// The k-th largest value of `v` (`k <= v.len()`), exactly. A branchless pass keeps the values
/// `>= mean + 1.2·std` (≈ 12 % of a k-WTA layer's pre-activations, more than the 5 % kept) in
/// `cand`, and a quickselect runs on those only; when fewer than `k` pass, on all of `v`.
#[inline]
fn kth_largest(v: &[f32], k: usize, cand: &mut [f32]) -> f32 {
    let n = v.len();
    let (mut s, mut s2) = (0f32, 0f32);
    for &x in v {
        s += x;
        s2 += x * x;
    }
    let mean = s / n as f32;
    let std = (s2 / n as f32 - mean * mean).max(0.0).sqrt();
    let t0 = mean + 1.2 * std;
    let cand = &mut cand[..n];
    let mut m = 0;
    for &x in v {
        cand[m] = x;
        m += (x >= t0) as usize;
    }
    if m < k {
        cand.copy_from_slice(v);
        m = n;
    }
    let c = &mut cand[..m];
    let (_, &mut kth, _) = c.select_nth_unstable_by(m - k, |a, b| a.total_cmp(b));
    kth
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same expectations as `sky-train`'s `s2::es` tests (keeps the two generators identical).
    #[test]
    fn projection_is_fixed_by_the_seed() {
        let p = projection(42, 64, 512, 8);
        assert_eq!(p.len(), 512 * 8);
        let idx: Vec<usize> = p[..8].iter().map(|x| x.0).collect();
        let sum: usize = p.iter().map(|x| x.0).sum();
        let neg = p.iter().filter(|x| x.1 < 0.0).count();
        assert_eq!(
            (idx, sum, neg),
            (vec![38, 51, 23, 63, 26, 58, 28, 46], 128_198, 2037)
        );
    }

    #[test]
    fn kth_largest_matches_sorting() {
        let mut s = 7u64;
        for n in [1usize, 5, 64, 512] {
            let v: Vec<f32> = (0..n).map(|_| (splitmix64(&mut s) >> 40) as f32).collect();
            let mut sorted = v.clone();
            sorted.sort_by(|a, b| b.total_cmp(a));
            for k in [1, n.div_ceil(20), n.div_ceil(3), n] {
                let mut cand = vec![0.0; n];
                assert_eq!(kth_largest(&v, k, &mut cand), sorted[k - 1], "n {n} k {k}");
            }
        }
    }

    #[test]
    fn forward_matches_the_dense_definition() {
        let (d, units, k) = (16, 128, 6);
        let spec = EsSpec {
            seed: 9,
            units,
            k,
            fan_in: 4,
        };
        let mut s = 1u64;
        let mut rnd = || (splitmix64(&mut s) >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
        let down: Vec<f32> = (0..d * units).map(|_| rnd()).collect();
        let h: Vec<f32> = (0..2 * d).map(|_| rnd() * 4.0).collect();
        let es = EsFfn::new(spec, d, &down);
        let mut x = vec![0.5f32; 2 * d];
        let (mut u, mut cand) = (vec![0.0; 2 * units], vec![0.0; units]);
        es.forward(&h, 2, &mut x, &mut u, &mut cand);
        let p = projection(9, d, units, 4);
        for r in 0..2 {
            let mut pre = vec![0f32; units];
            for (n, &(i, w)) in p.iter().enumerate() {
                pre[n / 4] += w * h[r * d + i];
            }
            let mut sorted = pre.clone();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let active: Vec<usize> = (0..units).filter(|&j| pre[j] >= sorted[k - 1]).collect();
            assert_eq!(active.len(), k);
            for o in 0..d {
                let want = 0.5
                    + active
                        .iter()
                        .map(|&j| down[o * units + j] * pre[j])
                        .sum::<f32>();
                assert!(
                    (x[r * d + o] - want).abs() < 1e-5,
                    "{} vs {want}",
                    x[r * d + o]
                );
            }
        }
    }
}

#[cfg(test)]
mod prof {
    use super::*;
    #[test]
    #[ignore]
    fn es_forward_timing() {
        let (d, units, k, rows) = (64, 512, 26, 9);
        let spec = EsSpec {
            seed: 2026,
            units,
            k,
            fan_in: 8,
        };
        let mut s = 3u64;
        let mut rnd = || (splitmix64(&mut s) >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
        let down: Vec<f32> = (0..d * units).map(|_| rnd() * 0.1).collect();
        let h: Vec<f32> = (0..rows * d).map(|_| rnd() * 3.0).collect();
        let es = EsFfn::new(spec, d, &down);
        let mut x = vec![0f32; rows * d];
        let (mut u, mut top) = (vec![0.0; rows * units], vec![0.0; units]);
        let n = 20000;
        let t = std::time::Instant::now();
        for _ in 0..n {
            es.forward(&h, rows, &mut x, &mut u, &mut top);
        }
        eprintln!(
            "forward: {:.2} us",
            t.elapsed().as_secs_f64() * 1e6 / n as f64
        );
        let t = std::time::Instant::now();
        for _ in 0..n {
            es.up.matmul(&h, d, rows, None, &mut u, units, false);
        }
        eprintln!("up: {:.2} us", t.elapsed().as_secs_f64() * 1e6 / n as f64);
        let t = std::time::Instant::now();
        let mut acc = 0.0;
        for _ in 0..n {
            for r in 0..rows {
                acc += kth_largest(&u[r * units..(r + 1) * units], k, &mut top);
            }
        }
        eprintln!(
            "select: {:.2} us ({acc})",
            t.elapsed().as_secs_f64() * 1e6 / n as f64
        );
    }
}
