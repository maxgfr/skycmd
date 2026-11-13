//! The malloc heap of this process: what the Rust side and MLX's graphs hold outside the GPU
//! buffers (`mlx_rs::memory` counts those).

/// Blocks and bytes in use in every malloc zone (zeros where the system has no such count).
pub fn in_use() -> (u64, u64) {
    #[cfg(target_os = "macos")]
    {
        #[repr(C)]
        #[derive(Default)]
        struct Stats {
            blocks_in_use: u32,
            size_in_use: usize,
            max_size_in_use: usize,
            size_allocated: usize,
        }
        extern "C" {
            fn malloc_zone_statistics(zone: *mut std::ffi::c_void, stats: *mut Stats);
        }
        let mut s = Stats::default();
        // SAFETY: a null zone asks for the sum over all zones; `s` has the layout of
        // `malloc_statistics_t` (<malloc/malloc.h>).
        unsafe { malloc_zone_statistics(std::ptr::null_mut(), &mut s) };
        (u64::from(s.blocks_in_use), s.size_in_use as u64)
    }
    #[cfg(not(target_os = "macos"))]
    {
        (0, 0)
    }
}

/// Megabytes in use (decimal, as the `mlx_*_mb` fields of the logs).
pub fn mb() -> f64 {
    in_use().1 as f64 / 1e6
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(target_os = "macos")]
    fn counts_a_live_allocation() {
        let before = super::in_use();
        assert!(before.0 > 0 && before.1 > 0);
        // 64 MB held: other tests allocate at the same time, far less than that.
        let held = vec![1u8; 64 << 20];
        let after = super::in_use();
        assert!(after.1 > before.1 + (32 << 20), "{before:?} -> {after:?}");
        drop(held);
    }
}
