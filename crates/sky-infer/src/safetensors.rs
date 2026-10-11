//! Minimal safetensors reader (F32 / F16 / BF16 tensors).

use crate::json::Json;
use crate::{Error, Result};

/// One tensor entry of a safetensors file.
#[derive(Clone, Debug)]
pub struct StTensor<'a> {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data: &'a [u8],
}

impl<'a> StTensor<'a> {
    /// Values converted to f32.
    pub fn to_f32(&self) -> Result<Vec<f32>> {
        Ok(match self.dtype.as_str() {
            "F32" => self
                .data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            "F16" => self
                .data
                .chunks_exact(2)
                .map(|c| sky_gguf::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            "BF16" => self
                .data
                .chunks_exact(2)
                .map(|c| sky_gguf::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            other => {
                return Err(Error::Unsupported(format!(
                    "safetensors dtype {other} ({})",
                    self.name
                )))
            }
        })
    }
}

/// A parsed safetensors file borrowing its bytes.
#[derive(Clone, Debug)]
pub struct SafeTensors<'a> {
    pub tensors: Vec<StTensor<'a>>,
    pub metadata: Vec<(String, String)>,
}

impl<'a> SafeTensors<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(Error::Format("safetensors file too short".into()));
        }
        let n = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as usize;
        let header = bytes
            .get(8..8usize.saturating_add(n))
            .ok_or_else(|| Error::Format("safetensors header past the end".into()))?;
        let header = std::str::from_utf8(header)
            .map_err(|_| Error::Format("safetensors header is not UTF-8".into()))?;
        let j = Json::parse(header)?;
        let data = &bytes[8 + n..];
        let mut tensors = Vec::new();
        let mut metadata = Vec::new();
        for (name, v) in j
            .as_object()
            .ok_or_else(|| Error::Format("safetensors header is not an object".into()))?
        {
            if name == "__metadata__" {
                for (k, val) in v.as_object().unwrap_or(&[]) {
                    metadata.push((k.clone(), val.as_str().unwrap_or_default().to_string()));
                }
                continue;
            }
            let dtype = v
                .get("dtype")
                .and_then(|d| d.as_str())
                .unwrap_or_default()
                .to_string();
            let shape: Vec<usize> = v
                .get("shape")
                .and_then(|s| s.as_array())
                .ok_or_else(|| Error::Format(format!("{name}: no shape")))?
                .iter()
                .map(|d| d.as_u64().map(|d| d as usize))
                .collect::<Option<_>>()
                .ok_or_else(|| Error::Format(format!("{name}: bad shape")))?;
            let off = v
                .get("data_offsets")
                .and_then(|o| o.as_array())
                .filter(|o| o.len() == 2)
                .ok_or_else(|| Error::Format(format!("{name}: no data_offsets")))?;
            let (s, e) = (
                off[0].as_u64().unwrap_or(u64::MAX) as usize,
                off[1].as_u64().unwrap_or(0) as usize,
            );
            if s > e || e > data.len() {
                return Err(Error::Format(format!("{name}: data offsets out of range")));
            }
            let elem = match dtype.as_str() {
                "F64" | "I64" | "U64" => 8,
                "F32" | "I32" | "U32" => 4,
                "F16" | "BF16" | "I16" | "U16" => 2,
                _ => 1,
            };
            let size = shape
                .iter()
                .try_fold(elem as usize, |acc: usize, &d| acc.checked_mul(d));
            if size != Some(e - s) {
                return Err(Error::Format(format!(
                    "{name}: size does not match shape {shape:?} {dtype}"
                )));
            }
            tensors.push(StTensor {
                name: name.clone(),
                dtype,
                shape,
                data: &data[s..e],
            });
        }
        Ok(SafeTensors { tensors, metadata })
    }

    pub fn get(&self, name: &str) -> Option<&StTensor<'a>> {
        self.tensors.iter().find(|t| t.name == name)
    }
}
