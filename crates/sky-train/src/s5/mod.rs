//! s5: decision models fine-tuned from pretrained encoders (Ettin, mmBERT).
//!
//! - [`torch`]: reader of PyTorch `pytorch_model.bin` checkpoints;
//! - [`hf`]: `sky-train import-hf`, a Hugging Face ModernBERT checkpoint as a sky-train one;
//! - [`record`]: s5 records, whose options are `(id, definition)` pairs;
//! - [`score`]: `sky-train score`, teacher soft labels from Laya-format models;
//! - [`data`], [`loss`], [`train`]: `sky-train s5-train`, the fine-tuning recipe;
//! - [`iso`]: variant B, invariant by construction (segment mask, shared positions).

pub mod data;
pub mod hf;
pub mod iso;
pub mod loss;
pub mod record;
pub mod score;
pub mod torch;
pub mod train;
