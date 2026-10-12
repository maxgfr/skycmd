//! Expand-and-sparsify feed-forward (SPEC-S3 §9 ablation, config `ffn_es`, off by default).
//!
//! Like the fly mushroom body (≈50 projection neurons → ≈2,500 Kenyon cells, ≈5 % active), the
//! encoder FFN becomes `x + W_down · kWTA(P · LN(x))`:
//! * `P` (`units × d`, `units = expand · d`) is a fixed sparse random ±1 matrix: unit `j` reads
//!   `fan_in` distinct inputs with random signs, scaled by `1/√fan_in`. It is generated from a u64
//!   seed (the only thing the GGUF stores: 0 bytes of weights);
//! * k-WTA keeps the `k = round(k_frac · units)` largest pre-activations of each token (every unit
//!   `≥` the k-th largest value, so ties keep more) and zeroes the others;
//! * the learned `ffn_down` (`d × units`, ternary like every encoder matrix) reads the active units.
//!
//! The generator below is duplicated bit for bit in `sky-infer` (`s2::es`); the golden files and
//! the fixed-output tests of both crates keep them in sync.

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

/// The projection as a dense row-major `[units][d]` matrix (training).
pub fn dense(seed: u64, d: usize, units: usize, fan_in: usize) -> Vec<f32> {
    let mut m = vec![0f32; units * d];
    for (n, (i, w)) in projection(seed, d, units, fan_in).into_iter().enumerate() {
        m[(n / fan_in) * d + i] = w;
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same expectations as `sky-infer`'s `s2::es` tests (keeps the two generators identical).
    #[test]
    fn projection_is_fixed_by_the_seed() {
        let p = projection(42, 64, 512, 8);
        assert_eq!(p.len(), 512 * 8);
        let idx: Vec<usize> = p[..8].iter().map(|x| x.0).collect();
        let sum: usize = p.iter().map(|x| x.0).sum();
        let neg = p.iter().filter(|x| x.1 < 0.0).count();
        assert_eq!((idx, sum, neg), (ES_FIRST.to_vec(), ES_SUM, ES_NEG));
        for row in p.chunks(8) {
            for (a, x) in row.iter().enumerate() {
                assert!(row[..a].iter().all(|y| y.0 != x.0));
                assert!((x.1.abs() - 1.0 / 8f32.sqrt()).abs() < 1e-7);
            }
        }
        let m = dense(42, 64, 512, 8);
        assert_eq!(m.iter().filter(|&&v| v != 0.0).count(), 512 * 8);
        assert_ne!(projection(43, 64, 512, 8), p);
    }

    // Computed independently (a Python port of the generator).
    pub const ES_FIRST: [usize; 8] = [38, 51, 23, 63, 26, 58, 28, 46];
    pub const ES_SUM: usize = 128_198;
    pub const ES_NEG: usize = 2037;
}
