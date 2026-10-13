//! s5: decision models fine-tuned from pretrained encoders (Ettin, mmBERT).
//!
//! - [`torch`]: reader of PyTorch `pytorch_model.bin` checkpoints;
//! - [`hf`]: `sky-train import-hf`, a Hugging Face ModernBERT checkpoint as a sky-train one;
//! - [`record`]: s5 records, whose options are `(id, definition)` pairs;
//! - [`score`]: `sky-train score`, teacher soft labels from Laya-format models.

pub mod hf;
pub mod record;
pub mod score;
pub mod torch;
