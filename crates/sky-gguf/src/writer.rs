//! GGUF v3 writer (builder API).

use std::io::Write;

use crate::dtype::GgmlType;
use crate::reader::Value;
use crate::{Error, Result, ValueType, DEFAULT_ALIGNMENT, GGUF_MAGIC};

/// An owned metadata value, used to build files.
#[derive(Clone, Debug, PartialEq)]
pub enum OwnedValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    /// Elements must all have the same type (checked when written).
    Array(Vec<OwnedValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl OwnedValue {
    pub fn value_type(&self) -> ValueType {
        match self {
            OwnedValue::U8(_) => ValueType::U8,
            OwnedValue::I8(_) => ValueType::I8,
            OwnedValue::U16(_) => ValueType::U16,
            OwnedValue::I16(_) => ValueType::I16,
            OwnedValue::U32(_) => ValueType::U32,
            OwnedValue::I32(_) => ValueType::I32,
            OwnedValue::F32(_) => ValueType::F32,
            OwnedValue::Bool(_) => ValueType::Bool,
            OwnedValue::String(_) => ValueType::String,
            OwnedValue::Array(_) => ValueType::Array,
            OwnedValue::U64(_) => ValueType::U64,
            OwnedValue::I64(_) => ValueType::I64,
            OwnedValue::F64(_) => ValueType::F64,
        }
    }

    /// Array of strings.
    pub fn strings<S: AsRef<str>>(items: &[S]) -> Self {
        OwnedValue::Array(
            items
                .iter()
                .map(|s| OwnedValue::String(s.as_ref().to_string()))
                .collect(),
        )
    }

    /// Copy a borrowed value.
    pub fn from_value(v: &Value<'_>) -> Self {
        match v {
            Value::U8(x) => OwnedValue::U8(*x),
            Value::I8(x) => OwnedValue::I8(*x),
            Value::U16(x) => OwnedValue::U16(*x),
            Value::I16(x) => OwnedValue::I16(*x),
            Value::U32(x) => OwnedValue::U32(*x),
            Value::I32(x) => OwnedValue::I32(*x),
            Value::F32(x) => OwnedValue::F32(*x),
            Value::Bool(x) => OwnedValue::Bool(*x),
            Value::String(x) => OwnedValue::String(x.to_string()),
            Value::Array(a) => {
                OwnedValue::Array(a.iter().map(|e| OwnedValue::from_value(&e)).collect())
            }
            Value::U64(x) => OwnedValue::U64(*x),
            Value::I64(x) => OwnedValue::I64(*x),
            Value::F64(x) => OwnedValue::F64(*x),
        }
    }
}

macro_rules! impl_from {
    ($($t:ty => $v:ident),*) => {$(
        impl From<$t> for OwnedValue {
            fn from(x: $t) -> Self { OwnedValue::$v(x) }
        }
    )*};
}
impl_from!(u8 => U8, i8 => I8, u16 => U16, i16 => I16, u32 => U32, i32 => I32, f32 => F32,
           bool => Bool, String => String, u64 => U64, i64 => I64, f64 => F64);

impl From<&str> for OwnedValue {
    fn from(x: &str) -> Self {
        OwnedValue::String(x.to_string())
    }
}

struct PendingTensor {
    name: String,
    dims: Vec<u64>,
    dtype: GgmlType,
    data: Vec<u8>,
}

/// Builds a GGUF v3 file in memory.
///
/// ```
/// use sky_gguf::{GgufWriter, GgmlType, Gguf};
/// let mut w = GgufWriter::new();
/// w.add_kv("general.architecture", "demo");
/// // a PyTorch [2, 3] matrix: ggml dims are [3, 2] (ne[0] = inner dimension)
/// let vals: Vec<f32> = (0..6).map(|v| v as f32).collect();
/// w.add_tensor_f32("w", &[3, 2], GgmlType::F32, &vals).unwrap();
/// let bytes = w.to_bytes().unwrap();
/// let g = Gguf::parse(&bytes).unwrap();
/// assert_eq!(g.tensor("w").unwrap().shape(), vec![2, 3]);
/// ```
pub struct GgufWriter {
    alignment: u64,
    kvs: Vec<(String, OwnedValue)>,
    tensors: Vec<PendingTensor>,
}

impl Default for GgufWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl GgufWriter {
    /// New writer with the default alignment (32); `general.alignment` is always written.
    pub fn new() -> Self {
        GgufWriter {
            alignment: DEFAULT_ALIGNMENT,
            kvs: Vec::new(),
            tensors: Vec::new(),
        }
    }

    /// Set a custom data alignment (power of two).
    pub fn with_alignment(mut self, alignment: u64) -> Self {
        assert!(
            alignment.is_power_of_two(),
            "alignment must be a power of two"
        );
        self.alignment = alignment;
        self
    }

    /// Add (or replace) a metadata key.
    pub fn add_kv(&mut self, key: impl Into<String>, value: impl Into<OwnedValue>) -> &mut Self {
        let key = key.into();
        let value = value.into();
        if let Some(slot) = self.kvs.iter_mut().find(|(k, _)| *k == key) {
            slot.1 = value;
        } else {
            self.kvs.push((key, value));
        }
        self
    }

    /// Add a tensor from raw bytes already encoded as `dtype`. `dims` are in ggml order
    /// (`dims[0]` = innermost dimension).
    pub fn add_tensor(
        &mut self,
        name: impl Into<String>,
        dims: &[u64],
        dtype: GgmlType,
        bytes: Vec<u8>,
    ) -> Result<&mut Self> {
        let name = name.into();
        let n: u64 = dims.iter().product();
        let expected = dtype.byte_size(n)?;
        if dims.is_empty() || expected != bytes.len() as u64 {
            return Err(Error::Format(format!(
                "tensor {name}: {} bytes given, {expected} expected for {dims:?} {}",
                bytes.len(),
                dtype.name()
            )));
        }
        if dtype == GgmlType::Q8_0 && !dims[0].is_multiple_of(crate::QK8_0 as u64) {
            return Err(Error::Format(format!(
                "tensor {name}: Q8_0 rows must be a multiple of 32"
            )));
        }
        if self.tensors.iter().any(|t| t.name == name) {
            return Err(Error::Format(format!("duplicate tensor {name}")));
        }
        self.tensors.push(PendingTensor {
            name,
            dims: dims.to_vec(),
            dtype,
            data: bytes,
        });
        Ok(self)
    }

    /// Add a tensor from f32 values, encoded as `dtype` (F32, F16, BF16 or Q8_0).
    pub fn add_tensor_f32(
        &mut self,
        name: impl Into<String>,
        dims: &[u64],
        dtype: GgmlType,
        values: &[f32],
    ) -> Result<&mut Self> {
        let bytes = crate::dtype::encode_f32(dtype, values)?;
        self.add_tensor(name, dims, dtype, bytes)
    }

    pub fn n_tensors(&self) -> usize {
        self.tensors.len()
    }

    /// Serialize the whole file.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        self.write_to(&mut out)?;
        Ok(out)
    }

    /// Serialize the whole file to a writer.
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        let mut head = Vec::new();
        head.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        head.extend_from_slice(&3u32.to_le_bytes());
        head.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());

        let has_alignment = self.kvs.iter().any(|(k, _)| k == "general.alignment");
        let n_kv = self.kvs.len() + usize::from(!has_alignment);
        head.extend_from_slice(&(n_kv as u64).to_le_bytes());
        if !has_alignment {
            put_string(&mut head, "general.alignment");
            head.extend_from_slice(&ValueType::U32.to_u32().to_le_bytes());
            head.extend_from_slice(&(self.alignment as u32).to_le_bytes());
        }
        for (k, v) in &self.kvs {
            if k == "general.alignment" && v.clone() != OwnedValue::U32(self.alignment as u32) {
                return Err(Error::Format(
                    "general.alignment must match the writer alignment".into(),
                ));
            }
            put_string(&mut head, k);
            head.extend_from_slice(&v.value_type().to_u32().to_le_bytes());
            put_value(&mut head, v)?;
        }

        let mut offset = 0u64;
        let mut offsets = Vec::with_capacity(self.tensors.len());
        for t in &self.tensors {
            put_string(&mut head, &t.name);
            head.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
            for d in &t.dims {
                head.extend_from_slice(&d.to_le_bytes());
            }
            head.extend_from_slice(&t.dtype.to_u32().to_le_bytes());
            head.extend_from_slice(&offset.to_le_bytes());
            offsets.push(offset);
            offset = (offset + t.data.len() as u64).div_ceil(self.alignment) * self.alignment;
        }
        pad_to(&mut head, self.alignment);
        w.write_all(&head)?;

        let zeros = vec![0u8; self.alignment as usize];
        for t in &self.tensors {
            w.write_all(&t.data)?;
            let pad = (t.data.len() as u64).div_ceil(self.alignment) * self.alignment
                - t.data.len() as u64;
            w.write_all(&zeros[..pad as usize])?;
        }
        Ok(())
    }
}

fn pad_to(buf: &mut Vec<u8>, alignment: u64) {
    let len = buf.len() as u64;
    let target = len.div_ceil(alignment) * alignment;
    buf.resize(target as usize, 0);
}

fn put_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn put_value(buf: &mut Vec<u8>, v: &OwnedValue) -> Result<()> {
    match v {
        OwnedValue::U8(x) => buf.push(*x),
        OwnedValue::I8(x) => buf.push(*x as u8),
        OwnedValue::U16(x) => buf.extend_from_slice(&x.to_le_bytes()),
        OwnedValue::I16(x) => buf.extend_from_slice(&x.to_le_bytes()),
        OwnedValue::U32(x) => buf.extend_from_slice(&x.to_le_bytes()),
        OwnedValue::I32(x) => buf.extend_from_slice(&x.to_le_bytes()),
        OwnedValue::F32(x) => buf.extend_from_slice(&x.to_le_bytes()),
        OwnedValue::Bool(x) => buf.push(u8::from(*x)),
        OwnedValue::String(s) => put_string(buf, s),
        OwnedValue::U64(x) => buf.extend_from_slice(&x.to_le_bytes()),
        OwnedValue::I64(x) => buf.extend_from_slice(&x.to_le_bytes()),
        OwnedValue::F64(x) => buf.extend_from_slice(&x.to_le_bytes()),
        OwnedValue::Array(items) => {
            // an empty array is written as an array of u8 (any type is valid)
            let et = items
                .first()
                .map(|e| e.value_type())
                .unwrap_or(ValueType::U8);
            if items.iter().any(|e| e.value_type() != et) {
                return Err(Error::Format(
                    "array elements must all have the same type".into(),
                ));
            }
            buf.extend_from_slice(&et.to_u32().to_le_bytes());
            buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for e in items {
                put_value(buf, e)?;
            }
        }
    }
    Ok(())
}
