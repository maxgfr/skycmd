//! skycmd data pipeline: Hugging Face datasets → decision / MLM records (SPEC formats),
//! byte-level BPE tokenizer training, and the GLM teacher client.

pub mod csv;
pub mod decisions;
pub mod hf;
pub mod minhash;
pub mod npz;
pub mod plan;
pub mod record;
pub mod rng;
pub mod stats;
pub mod teacher;
pub mod text;
pub mod tok;

pub use record::{DecisionRecord, MlmRecord, QType, Question};
