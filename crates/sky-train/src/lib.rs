//! Training of the skycmd-s1 decision models from scratch with mlx-rs:
//! ModernBERT encoder + Laya decision head, MLM pretraining, decision
//! training (AdamW + Muon, WSD), temperature calibration, evaluation and
//! export to the oscar / HF directory layout.

pub mod config;
pub mod decision;
pub mod export;
pub mod metrics;
pub mod mlm;
pub mod model;
pub mod optim;
pub mod params;
pub mod rng;
pub mod sequence;
pub mod session;
pub mod train;

#[cfg(test)]
pub(crate) mod test_util {
    use std::sync::{Mutex, MutexGuard};

    static LOCK: Mutex<()> = Mutex::new(());

    /// Serializes tests that run MLX graphs (the Metal device is shared).
    pub fn lock() -> MutexGuard<'static, ()> {
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }
}
