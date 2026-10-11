//! The skycmd-s2 architecture (docs/SPEC-S2.md): two-stream static/state
//! encoders with native numbers, early exits, a cross-attention fusion head
//! and ternary QAT, with its training drivers, calibration, evaluation and
//! GGUF export.

pub mod config;
pub mod data;
pub mod eval;
pub mod gguf;
pub mod golden;
pub mod model;
pub mod train;
