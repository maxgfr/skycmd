//! Small deterministic PRNG and stable hashing.
//!
//! Everything in the data pipeline must be reproducible from a seed and must not
//! depend on the platform or on `std`'s randomized hasher, so we ship our own.

/// SplitMix64 generator: tiny, fast, and good enough for shuffling and sampling.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    /// Derives an independent stream from a seed and a label (e.g. a source name).
    pub fn derived(seed: u64, label: &str) -> Self {
        Self::new(seed ^ hash64(label.as_bytes()))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix64(self.state)
    }

    /// Uniform integer in `0..n` (n > 0), using rejection to avoid modulo bias.
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "below(0)");
        let n = n as u64;
        let zone = u64::MAX - (u64::MAX % n);
        loop {
            let x = self.next_u64();
            if x < zone {
                return (x % n) as usize;
            }
        }
    }

    /// Uniform float in `[0, 1)`.
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }
}

/// SplitMix64 finalizer, a strong 64-bit mixing function.
#[inline]
pub fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Stable 64-bit hash (FNV-1a followed by a SplitMix finalizer).
pub fn hash64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    mix64(h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_in_range() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        for _ in 0..1000 {
            let x = a.below(13);
            assert_eq!(x, b.below(13));
            assert!(x < 13);
            let f = a.f64();
            assert_eq!(f, b.f64());
            assert!((0.0..1.0).contains(&f));
        }
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut v: Vec<u32> = (0..100).collect();
        Rng::new(1).shuffle(&mut v);
        assert_ne!(v, (0..100).collect::<Vec<_>>());
        let mut s = v.clone();
        s.sort();
        assert_eq!(s, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn hash_is_stable() {
        assert_eq!(hash64(b"abc"), hash64(b"abc"));
        assert_ne!(hash64(b"abc"), hash64(b"abd"));
        assert_ne!(
            Rng::derived(1, "a").next_u64(),
            Rng::derived(1, "b").next_u64()
        );
    }
}
