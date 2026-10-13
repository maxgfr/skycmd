//! A tiny deterministic generator (xoshiro256** seeded through SplitMix64, the one of
//! `sky-train`), so resamples and option draws are the same on every platform and run.

/// xoshiro256** seeded through SplitMix64.
#[derive(Debug, Clone, PartialEq, Eq)]
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

/// FNV-1a hash of a string, for seeding per-item generators.
pub fn hash_str(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)
    })
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_is_pinned() {
        // xoshiro256** after SplitMix64(0), checked against an independent implementation.
        // Changing these values changes every published interval.
        let mut r = Rng::new(0);
        let got: Vec<u64> = (0..3).map(|_| r.next_u64()).collect();
        assert_eq!(
            got,
            [
                11_091_344_671_253_066_420,
                13_793_997_310_169_335_082,
                1_900_383_378_846_508_768
            ]
        );
        assert_eq!(hash_str(""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(hash_str("a"), 0xaf63_dc4c_8601_ec8c);
    }

    #[test]
    fn below_f64_and_shuffle() {
        let mut r = Rng::new(1);
        let mut seen = [0usize; 5];
        for _ in 0..5000 {
            let x = r.f64();
            assert!((0.0..1.0).contains(&x));
            seen[r.below(5)] += 1;
        }
        assert!(seen.iter().all(|&c| c > 800));
        let mut v: Vec<usize> = (0..10).collect();
        r.shuffle(&mut v);
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
        assert_ne!(v, sorted);
    }
}
