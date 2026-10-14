//! `sky-gguf`: a small, dependency-free GGUF v3 reader and writer.
//!
//! * [`Gguf::parse`] reads a file held in memory without copying tensor data (metadata of all
//!   types including nested arrays, tensor infos, aligned data offsets).
//! * [`GgufWriter`] builds files (`general.alignment` = 32 by default).
//! * Tensor types F32, F16, BF16, Q8_0, Q4_0 and I8 can be converted to and from f32
//!   ([`dequantize`], [`encode_f32`], [`quantize_q8_0`], [`quantize_q4_0`]).
//!
//! **Dimension order.** GGUF/ggml dimensions are listed innermost first: `ne[0]` is the
//! fastest-varying, contiguous dimension. A PyTorch tensor of shape `[rows, cols]` (row-major)
//! is stored with `dims = [cols, rows]` and identical bytes; [`TensorInfo::shape`] returns the
//! PyTorch order.

mod dtype;
mod reader;
mod writer;

pub use dtype::{
    bf16_to_f32, dequantize, dequantize_into, encode_f32, f16_to_f32, f32_to_bf16, f32_to_f16,
    q4_0_block, quantize_q4_0, quantize_q4_0_into, quantize_q8_0, quantize_q8_0_into, GgmlType,
    Q4_0_BLOCK_BYTES, Q8_0_BLOCK_BYTES, QK4_0, QK8_0,
};
pub use reader::{Array, ArrayIter, Gguf, TensorInfo, Value};
pub use writer::{GgufWriter, OwnedValue};

/// "GGUF" read as a little-endian u32.
pub const GGUF_MAGIC: u32 = 0x4655_4747;
/// Default tensor data alignment.
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// Metadata value types (`enum gguf_type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ValueType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    String,
    Array,
    U64,
    I64,
    F64,
}

impl ValueType {
    pub fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => ValueType::U8,
            1 => ValueType::I8,
            2 => ValueType::U16,
            3 => ValueType::I16,
            4 => ValueType::U32,
            5 => ValueType::I32,
            6 => ValueType::F32,
            7 => ValueType::Bool,
            8 => ValueType::String,
            9 => ValueType::Array,
            10 => ValueType::U64,
            11 => ValueType::I64,
            12 => ValueType::F64,
            _ => return Err(Error::Format(format!("unknown metadata value type {v}"))),
        })
    }

    pub fn to_u32(self) -> u32 {
        match self {
            ValueType::U8 => 0,
            ValueType::I8 => 1,
            ValueType::U16 => 2,
            ValueType::I16 => 3,
            ValueType::U32 => 4,
            ValueType::I32 => 5,
            ValueType::F32 => 6,
            ValueType::Bool => 7,
            ValueType::String => 8,
            ValueType::Array => 9,
            ValueType::U64 => 10,
            ValueType::I64 => 11,
            ValueType::F64 => 12,
        }
    }

    /// Encoded size of a scalar of this type (`None` for strings and arrays).
    pub fn fixed_size(self) -> Option<usize> {
        match self {
            ValueType::U8 | ValueType::I8 | ValueType::Bool => Some(1),
            ValueType::U16 | ValueType::I16 => Some(2),
            ValueType::U32 | ValueType::I32 | ValueType::F32 => Some(4),
            ValueType::U64 | ValueType::I64 | ValueType::F64 => Some(8),
            ValueType::String | ValueType::Array => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ValueType::U8 => "u8",
            ValueType::I8 => "i8",
            ValueType::U16 => "u16",
            ValueType::I16 => "i16",
            ValueType::U32 => "u32",
            ValueType::I32 => "i32",
            ValueType::F32 => "f32",
            ValueType::Bool => "bool",
            ValueType::String => "string",
            ValueType::Array => "array",
            ValueType::U64 => "u64",
            ValueType::I64 => "i64",
            ValueType::F64 => "f64",
        }
    }
}

/// Errors of this crate.
#[derive(Debug)]
pub enum Error {
    /// Unexpected end of data.
    Eof,
    /// Malformed file or invalid input.
    Format(String),
    /// Valid but not supported (version, tensor type).
    Unsupported(String),
    /// A required key or tensor is absent.
    Missing(String),
    Io(std::io::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Eof => write!(f, "unexpected end of GGUF data"),
            Error::Format(m) => write!(f, "invalid GGUF: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported GGUF feature: {m}"),
            Error::Missing(m) => write!(f, "missing in GGUF: {m}"),
            Error::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
