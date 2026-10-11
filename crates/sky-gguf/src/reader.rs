//! Zero-copy GGUF v3 reader.
//!
//! Dimension order: GGUF stores tensor dimensions in ggml order, where `dims[0]` (`ne[0]`) is the
//! fastest-varying (innermost, contiguous) dimension. A PyTorch tensor of shape `[out, in]` is
//! therefore stored with `dims = [in, out]`, with the same bytes (row-major `[out][in]`).

use crate::dtype::{dequantize, GgmlType};
use crate::{Error, Result, ValueType, DEFAULT_ALIGNMENT, GGUF_MAGIC};

/// A metadata value borrowed from the file bytes.
#[derive(Clone, Debug, PartialEq)]
pub enum Value<'a> {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(&'a str),
    Array(Array<'a>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl<'a> Value<'a> {
    pub fn value_type(&self) -> ValueType {
        match self {
            Value::U8(_) => ValueType::U8,
            Value::I8(_) => ValueType::I8,
            Value::U16(_) => ValueType::U16,
            Value::I16(_) => ValueType::I16,
            Value::U32(_) => ValueType::U32,
            Value::I32(_) => ValueType::I32,
            Value::F32(_) => ValueType::F32,
            Value::Bool(_) => ValueType::Bool,
            Value::String(_) => ValueType::String,
            Value::Array(_) => ValueType::Array,
            Value::U64(_) => ValueType::U64,
            Value::I64(_) => ValueType::I64,
            Value::F64(_) => ValueType::F64,
        }
    }

    /// Any integer value as i64 (u64 values above i64::MAX are rejected).
    pub fn as_i64(&self) -> Option<i64> {
        Some(match *self {
            Value::U8(v) => v as i64,
            Value::I8(v) => v as i64,
            Value::U16(v) => v as i64,
            Value::I16(v) => v as i64,
            Value::U32(v) => v as i64,
            Value::I32(v) => v as i64,
            Value::U64(v) => i64::try_from(v).ok()?,
            Value::I64(v) => v,
            Value::Bool(v) => v as i64,
            _ => return None,
        })
    }

    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U64(v) => Some(v),
            _ => self.as_i64().and_then(|v| u64::try_from(v).ok()),
        }
    }

    /// Any numeric value as f64.
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v as f64),
            Value::F64(v) => Some(v),
            _ => self.as_i64().map(|v| v as f64),
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&'a str> {
        match *self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Array<'a>> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
}

/// A typed metadata array. The elements stay in the file bytes and are decoded on access.
#[derive(Clone, Debug, PartialEq)]
pub struct Array<'a> {
    elem_type: ValueType,
    len: usize,
    /// Raw bytes of all elements.
    raw: &'a [u8],
}

impl<'a> Array<'a> {
    pub fn elem_type(&self) -> ValueType {
        self.elem_type
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Iterate over the elements as [`Value`]s.
    pub fn iter(&self) -> ArrayIter<'a> {
        ArrayIter {
            cur: Cursor {
                buf: self.raw,
                pos: 0,
            },
            elem_type: self.elem_type,
            left: self.len,
        }
    }

    /// Elements as strings (`None` if the array is not an array of strings).
    pub fn strings(&self) -> Option<Vec<&'a str>> {
        if self.elem_type != ValueType::String {
            return None;
        }
        self.iter().map(|v| v.as_str()).collect()
    }

    /// Integer elements as i64.
    pub fn to_i64_vec(&self) -> Option<Vec<i64>> {
        self.iter().map(|v| v.as_i64()).collect()
    }

    /// Numeric elements as f64.
    pub fn to_f64_vec(&self) -> Option<Vec<f64>> {
        self.iter().map(|v| v.as_f64()).collect()
    }
}

pub struct ArrayIter<'a> {
    cur: Cursor<'a>,
    elem_type: ValueType,
    left: usize,
}

impl<'a> Iterator for ArrayIter<'a> {
    type Item = Value<'a>;

    fn next(&mut self) -> Option<Value<'a>> {
        if self.left == 0 {
            return None;
        }
        self.left -= 1;
        // the array was fully validated when the file was parsed
        Some(
            self.cur
                .value(self.elem_type, 0)
                .expect("validated GGUF array"),
        )
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left, Some(self.left))
    }
}

/// Description of one tensor of the file.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorInfo<'a> {
    pub name: &'a str,
    /// ggml dimensions, `dims[0]` = fastest (innermost) dimension.
    pub dims: Vec<u64>,
    pub dtype: GgmlType,
    /// Offset of the data relative to the start of the tensor data section.
    pub offset: u64,
    /// Data size in bytes (0 if the type is unknown).
    pub size: u64,
}

impl<'a> TensorInfo<'a> {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }

    /// Shape in PyTorch / row-major order (outermost first), i.e. the reverse of `dims`.
    pub fn shape(&self) -> Vec<usize> {
        self.dims.iter().rev().map(|&d| d as usize).collect()
    }
}

/// A parsed GGUF file borrowing the underlying bytes.
#[derive(Clone, Debug)]
pub struct Gguf<'a> {
    pub version: u32,
    pub alignment: u64,
    pub kvs: Vec<(&'a str, Value<'a>)>,
    pub tensors: Vec<TensorInfo<'a>>,
    /// Absolute offset of the tensor data section in the file.
    pub data_offset: u64,
    bytes: &'a [u8],
}

impl<'a> Gguf<'a> {
    /// Parse a GGUF v2/v3 file held in memory. Tensor data are not copied.
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let mut c = Cursor { buf: bytes, pos: 0 };
        let magic = c.u32()?;
        if magic != GGUF_MAGIC {
            return Err(Error::Format(format!(
                "bad magic {magic:#010x}, not a GGUF file"
            )));
        }
        let version = c.u32()?;
        if !(2..=3).contains(&version) {
            return Err(Error::Unsupported(format!("GGUF version {version}")));
        }
        let n_tensors = c.u64()?;
        let n_kv = c.u64()?;
        // every kv / tensor takes at least 8 bytes: reject absurd counts before allocating
        let max_items = (bytes.len() / 8) as u64;
        if n_tensors > max_items || n_kv > max_items {
            return Err(Error::Format(
                "tensor or kv count larger than the file".into(),
            ));
        }

        let mut kvs = Vec::with_capacity(n_kv as usize);
        for _ in 0..n_kv {
            let key = c.string()?;
            let ty = ValueType::from_u32(c.u32()?)?;
            let val = c.value(ty, 0)?;
            kvs.push((key, val));
        }

        let mut alignment = DEFAULT_ALIGNMENT;
        if let Some((_, v)) = kvs.iter().find(|(k, _)| *k == "general.alignment") {
            alignment = v
                .as_u64()
                .ok_or_else(|| Error::Format("general.alignment is not an integer".into()))?;
            if alignment == 0 || !alignment.is_power_of_two() {
                return Err(Error::Format(format!("invalid alignment {alignment}")));
            }
        }

        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = c.string()?;
            let n_dims = c.u32()?;
            if n_dims > 8 {
                return Err(Error::Format(format!("tensor {name}: {n_dims} dimensions")));
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(c.u64()?);
            }
            let dtype = GgmlType::from_u32(c.u32()?);
            let offset = c.u64()?;
            if offset % alignment != 0 {
                return Err(Error::Format(format!(
                    "tensor {name}: offset {offset} is not aligned"
                )));
            }
            let n: u64 = dims
                .iter()
                .try_fold(1u64, |acc, &d| acc.checked_mul(d))
                .ok_or_else(|| Error::Format(format!("tensor {name}: dimensions overflow")))?;
            // unknown types have no size (the tensor can be listed, not read)
            let size = match dtype {
                GgmlType::Other(_) => 0,
                _ => dtype
                    .byte_size(n)
                    .map_err(|e| Error::Format(format!("tensor {name}: {e}")))?,
            };
            if dtype == GgmlType::Q8_0
                && dims
                    .first()
                    .is_some_and(|&d| !d.is_multiple_of(crate::QK8_0 as u64))
            {
                return Err(Error::Format(format!(
                    "tensor {name}: Q8_0 rows must be a multiple of 32"
                )));
            }
            tensors.push(TensorInfo {
                name,
                dims,
                dtype,
                offset,
                size,
            });
        }

        let data_offset = align_up(c.pos as u64, alignment);
        for t in &tensors {
            let end = data_offset
                .checked_add(t.offset)
                .and_then(|v| v.checked_add(t.size))
                .ok_or_else(|| Error::Format(format!("tensor {}: offset overflow", t.name)))?;
            if end > bytes.len() as u64 {
                return Err(Error::Format(format!(
                    "tensor {}: data past the end of the file",
                    t.name
                )));
            }
        }
        Ok(Gguf {
            version,
            alignment,
            kvs,
            tensors,
            data_offset,
            bytes,
        })
    }

    /// Metadata value of `key`.
    pub fn get(&self, key: &str) -> Option<&Value<'a>> {
        self.kvs.iter().find(|(k, _)| *k == key).map(|(_, v)| v)
    }

    pub fn get_str(&self, key: &str) -> Option<&'a str> {
        self.get(key).and_then(|v| v.as_str())
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(|v| v.as_u64())
    }

    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(|v| v.as_f64())
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.get(key).and_then(|v| v.as_bool())
    }

    pub fn get_array(&self, key: &str) -> Option<&Array<'a>> {
        self.get(key).and_then(|v| v.as_array())
    }

    /// The `general.architecture` value.
    pub fn architecture(&self) -> Option<&'a str> {
        self.get_str("general.architecture")
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo<'a>> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Raw bytes of a tensor (borrowed from the file).
    pub fn tensor_bytes(&self, info: &TensorInfo<'a>) -> &'a [u8] {
        let start = (self.data_offset + info.offset) as usize;
        &self.bytes[start..start + info.size as usize]
    }

    /// Tensor data converted to f32 (F32, F16, BF16 or Q8_0).
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self
            .tensor(name)
            .ok_or_else(|| Error::Missing(name.to_string()))?;
        dequantize(info.dtype, self.tensor_bytes(info))
    }
}

fn align_up(v: u64, a: u64) -> u64 {
    v.div_ceil(a) * a
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or(Error::Eof)?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn arr<const N: usize>(&mut self) -> Result<[u8; N]> {
        Ok(self.take(N)?.try_into().expect("length checked"))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.arr()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.arr()?))
    }

    fn string(&mut self) -> Result<&'a str> {
        let n = self.u64()?;
        let n = usize::try_from(n).map_err(|_| Error::Eof)?;
        let b = self.take(n)?;
        std::str::from_utf8(b).map_err(|_| Error::Format("string is not valid UTF-8".into()))
    }

    fn value(&mut self, ty: ValueType, depth: u32) -> Result<Value<'a>> {
        Ok(match ty {
            ValueType::U8 => Value::U8(self.arr::<1>()?[0]),
            ValueType::I8 => Value::I8(self.arr::<1>()?[0] as i8),
            ValueType::U16 => Value::U16(u16::from_le_bytes(self.arr()?)),
            ValueType::I16 => Value::I16(i16::from_le_bytes(self.arr()?)),
            ValueType::U32 => Value::U32(self.u32()?),
            ValueType::I32 => Value::I32(i32::from_le_bytes(self.arr()?)),
            ValueType::F32 => Value::F32(f32::from_le_bytes(self.arr()?)),
            ValueType::Bool => {
                let b = self.arr::<1>()?[0];
                if b > 1 {
                    return Err(Error::Format(format!("invalid bool {b}")));
                }
                Value::Bool(b == 1)
            }
            ValueType::String => Value::String(self.string()?),
            ValueType::U64 => Value::U64(self.u64()?),
            ValueType::I64 => Value::I64(i64::from_le_bytes(self.arr()?)),
            ValueType::F64 => Value::F64(f64::from_le_bytes(self.arr()?)),
            ValueType::Array => {
                if depth > 4 {
                    return Err(Error::Format("arrays nested too deeply".into()));
                }
                let elem_type = ValueType::from_u32(self.u32()?)?;
                let len = usize::try_from(self.u64()?).map_err(|_| Error::Eof)?;
                let start = self.pos;
                match elem_type.fixed_size() {
                    Some(sz) => {
                        let total = len.checked_mul(sz).ok_or(Error::Eof)?;
                        self.take(total)?;
                        if elem_type == ValueType::Bool {
                            let raw = &self.buf[start..self.pos];
                            if raw.iter().any(|&b| b > 1) {
                                return Err(Error::Format("invalid bool in array".into()));
                            }
                        }
                    }
                    None => {
                        if len > self.buf.len() - self.pos {
                            return Err(Error::Eof); // each element takes at least one byte
                        }
                        for _ in 0..len {
                            self.value(elem_type, depth + 1)?;
                        }
                    }
                }
                Value::Array(Array {
                    elem_type,
                    len,
                    raw: &self.buf[start..self.pos],
                })
            }
        })
    }
}
