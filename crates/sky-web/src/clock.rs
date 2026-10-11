//! A monotonic microsecond clock: `performance.now()` in the browser (window or worker),
//! `std::time::Instant` on native targets (tests, tools).

#[cfg(target_arch = "wasm32")]
mod imp {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = performance, js_name = now)]
        fn performance_now() -> f64;
    }

    pub fn now_us() -> f64 {
        performance_now() * 1000.0
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    use std::sync::OnceLock;
    use std::time::Instant;

    static START: OnceLock<Instant> = OnceLock::new();

    pub fn now_us() -> f64 {
        START.get_or_init(Instant::now).elapsed().as_secs_f64() * 1e6
    }
}

/// Microseconds since an arbitrary origin.
pub fn now_us() -> f64 {
    imp::now_us()
}

/// Median, 95th percentile and mean of a set of durations (all 0 when empty).
pub fn summarize(samples: &[f64]) -> (f64, f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut v = samples.to_vec();
    v.sort_by(f64::total_cmp);
    let pick = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    (pick(0.5), pick(0.95), mean)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_is_monotonic() {
        let a = now_us();
        let b = now_us();
        assert!(b >= a);
    }

    #[test]
    fn summary_of_known_values() {
        let v: Vec<f64> = (1..=100).map(f64::from).collect();
        let (med, p95, mean) = summarize(&v);
        assert!((med - 50.0).abs() <= 1.0, "{med}");
        assert!((p95 - 95.0).abs() <= 1.0, "{p95}");
        assert!((mean - 50.5).abs() < 1e-9);
        assert_eq!(summarize(&[]), (0.0, 0.0, 0.0));
    }
}
