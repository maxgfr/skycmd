//! `sky-infer`: a small, fast inference engine for Laya-format "System One" decision models
//! (ModernBERT encoder + typed decision head), for native targets and wasm32.

pub mod api;
pub mod convert;
pub mod engine;
pub mod json;
pub mod kernels;
pub mod model;
pub mod prompt;
pub mod quant;
pub mod safetensors;
pub mod tokenizer;
#[cfg(feature = "wasm")]
pub mod wasm;

pub use api::{Answer, DecisionRequest, DecisionResponse, Question};
pub use engine::{Engine, EngineOptions, ModelInfo};
pub use model::{Model, ModelConfig, QuantizeSpec};
pub use prompt::{QType, QuestionKind};
pub use tokenizer::{BpeCache, SpecialTokens, Tokenizer};

/// Errors of this crate.
#[derive(Debug)]
pub enum Error {
    /// Malformed input file or data.
    Format(String),
    /// Valid but unsupported feature.
    Unsupported(String),
    /// A required tensor, key or file is absent.
    Missing(String),
    /// Invalid decision request.
    InvalidRequest(String),
    Gguf(sky_gguf::Error),
    Io(std::io::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Format(m) => write!(f, "invalid format: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::Missing(m) => write!(f, "missing: {m}"),
            Error::InvalidRequest(m) => write!(f, "invalid request: {m}"),
            Error::Gguf(e) => write!(f, "{e}"),
            Error::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<sky_gguf::Error> for Error {
    fn from(e: sky_gguf::Error) -> Self {
        Error::Gguf(e)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
