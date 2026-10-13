//! skycmd data pipeline: Hugging Face datasets → decision / MLM / command records (SPEC
//! formats), byte-level BPE tokenizer training (s1, and s2 with native numbers), and the GLM
//! teacher client and its data integration.

pub mod augment;
pub mod commands;
pub mod csv;
pub mod decisions;
pub mod hf;
pub mod minhash;
pub mod nlu;
pub mod npz;
pub mod num;
pub mod plan;
pub mod record;
pub mod rng;
pub mod s4;
pub mod stats;
pub mod teacher;
pub mod text;
pub mod tok;
pub mod typed;

pub use record::{DecisionRecord, MlmRecord, QType, Question};
