//! A tiny deterministic RNG (xoshiro256**) whose whole state fits in a
//! checkpoint, so resumed runs draw exactly the same numbers.

use serde::{Deserialize, Serialize};

/// xoshiro256** seeded through SplitMix64.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rng {
    s: [u64; 4],
}

fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut x = seed;
        Self {
            s: [
                splitmix64(&mut x),
                splitmix64(&mut x),
                splitmix64(&mut x),
                splitmix64(&mut x),
            ],
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in `[0, 1)`.
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in `[0, n)`; `n` must be > 0.
    pub fn below(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }

    /// Fisher–Yates shuffle.
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }

    /// Index drawn proportionally to `weights` (all >= 0, sum > 0).
    pub fn weighted(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().sum();
        let mut r = self.f64() * total;
        for (i, w) in weights.iter().enumerate() {
            if r < *w {
                return i;
            }
            r -= w;
        }
        weights.len() - 1
    }

    /// A fresh seed for a derived generator or an MLX key.
    pub fn seed(&mut self) -> u64 {
        self.next_u64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_serializable() {
        let mut a = Rng::new(7);
        a.next_u64();
        let saved = serde_json::to_string(&a).unwrap();
        let mut b: Rng = serde_json::from_str(&saved).unwrap();
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn below_and_f64_in_range() {
        let mut r = Rng::new(1);
        let mut seen = [0usize; 5];
        for _ in 0..5000 {
            let x = r.f64();
            assert!((0.0..1.0).contains(&x));
            seen[r.below(5)] += 1;
        }
        assert!(seen.iter().all(|&c| c > 800));
    }
}
