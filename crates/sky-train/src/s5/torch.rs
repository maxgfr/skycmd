//! Reader of PyTorch `torch.save` checkpoints (`pytorch_model.bin`): a zip archive holding
//! `<prefix>/data.pkl` (the pickled state dict) and one raw little-endian file per storage
//! (`<prefix>/data/<key>`). Only what a state dict of plain tensors needs is supported: the
//! pickle opcodes of protocols 2–4 that build dicts, tuples, strings and ints, the
//! `torch._utils._rebuild_tensor_v2` reducer and `collections.OrderedDict`. Nothing is ever
//! executed: unknown globals become inert values and a non-tensor entry is an error.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};

/// Element type of a storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TorchDtype {
    F32,
    F16,
    BF16,
}

impl TorchDtype {
    fn size(self) -> usize {
        match self {
            TorchDtype::F32 => 4,
            TorchDtype::F16 | TorchDtype::BF16 => 2,
        }
    }

    fn from_storage(name: &str) -> Result<Self> {
        Ok(match name {
            "FloatStorage" => TorchDtype::F32,
            "HalfStorage" => TorchDtype::F16,
            "BFloat16Storage" => TorchDtype::BF16,
            other => bail!("unsupported storage type torch.{other}"),
        })
    }
}

/// One tensor of the state dict, converted to f32 in row-major order.
#[derive(Debug, Clone, PartialEq)]
pub struct TorchTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
enum Val {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Tuple(Vec<Val>),
    List(Vec<Val>),
    Dict(Vec<(Val, Val)>),
    Global(String, String),
    Storage {
        dtype: TorchDtype,
        key: String,
    },
    Tensor {
        dtype: TorchDtype,
        key: String,
        offset: usize,
        shape: Vec<usize>,
        stride: Vec<usize>,
    },
    /// The result of a reducer we do not model (e.g. backward hooks): kept inert.
    Opaque,
    Mark,
}

fn as_usize(v: &Val) -> Result<usize> {
    match v {
        Val::Int(i) if *i >= 0 => Ok(*i as usize),
        other => bail!("expected a non-negative int, got {other:?}"),
    }
}

fn as_usizes(v: &Val) -> Result<Vec<usize>> {
    match v {
        Val::Tuple(t) | Val::List(t) => t.iter().map(as_usize).collect(),
        other => bail!("expected a tuple of ints, got {other:?}"),
    }
}

/// Runs the pickle program and returns the top-level value.
fn unpickle(data: &[u8]) -> Result<Val> {
    let mut stack: Vec<Val> = Vec::new();
    let mut memo: HashMap<u64, Val> = HashMap::new();
    let mut i = 0usize;
    let take = |i: &mut usize, n: usize| -> Result<&[u8]> {
        let s = data
            .get(*i..*i + n)
            .ok_or_else(|| anyhow!("truncated pickle at byte {}", *i))?;
        *i += n;
        Ok(s)
    };
    fn pop_mark(stack: &mut Vec<Val>) -> Result<Vec<Val>> {
        let at = stack
            .iter()
            .rposition(|v| *v == Val::Mark)
            .ok_or_else(|| anyhow!("pickle MARK not found"))?;
        let items = stack.split_off(at + 1);
        stack.pop();
        Ok(items)
    }
    let pop = |stack: &mut Vec<Val>| stack.pop().ok_or_else(|| anyhow!("pickle stack underflow"));
    loop {
        let op = *take(&mut i, 1)?.first().unwrap();
        match op {
            0x80 => {
                take(&mut i, 1)?; // PROTO
            }
            0x95 => {
                take(&mut i, 8)?; // FRAME
            }
            b'.' => return pop(&mut stack), // STOP
            b'}' => stack.push(Val::Dict(Vec::new())),
            b']' => stack.push(Val::List(Vec::new())),
            b')' => stack.push(Val::Tuple(Vec::new())),
            b'(' => stack.push(Val::Mark),
            b'N' => stack.push(Val::None),
            0x88 => stack.push(Val::Bool(true)),
            0x89 => stack.push(Val::Bool(false)),
            b'q' => {
                let k = take(&mut i, 1)?[0] as u64;
                memo.insert(
                    k,
                    stack
                        .last()
                        .cloned()
                        .ok_or_else(|| anyhow!("BINPUT on empty stack"))?,
                );
            }
            b'r' => {
                let k = u32::from_le_bytes(take(&mut i, 4)?.try_into()?) as u64;
                memo.insert(
                    k,
                    stack
                        .last()
                        .cloned()
                        .ok_or_else(|| anyhow!("LONG_BINPUT on empty stack"))?,
                );
            }
            0x94 => {
                let k = memo.len() as u64; // MEMOIZE
                memo.insert(
                    k,
                    stack
                        .last()
                        .cloned()
                        .ok_or_else(|| anyhow!("MEMOIZE on empty stack"))?,
                );
            }
            b'h' => {
                let k = take(&mut i, 1)?[0] as u64;
                stack.push(
                    memo.get(&k)
                        .cloned()
                        .ok_or_else(|| anyhow!("BINGET {k} unknown"))?,
                );
            }
            b'j' => {
                let k = u32::from_le_bytes(take(&mut i, 4)?.try_into()?) as u64;
                stack.push(
                    memo.get(&k)
                        .cloned()
                        .ok_or_else(|| anyhow!("LONG_BINGET {k} unknown"))?,
                );
            }
            b'X' => {
                let n = u32::from_le_bytes(take(&mut i, 4)?.try_into()?) as usize;
                stack.push(Val::Str(String::from_utf8(take(&mut i, n)?.to_vec())?));
            }
            0x8c => {
                let n = take(&mut i, 1)?[0] as usize;
                stack.push(Val::Str(String::from_utf8(take(&mut i, n)?.to_vec())?));
            }
            b'K' => stack.push(Val::Int(take(&mut i, 1)?[0] as i64)),
            b'M' => stack.push(Val::Int(
                u16::from_le_bytes(take(&mut i, 2)?.try_into()?) as i64
            )),
            b'J' => stack.push(Val::Int(
                i32::from_le_bytes(take(&mut i, 4)?.try_into()?) as i64
            )),
            0x8a => {
                // LONG1: little-endian two's complement of n bytes.
                let n = take(&mut i, 1)?[0] as usize;
                let b = take(&mut i, n)?;
                if n > 8 {
                    bail!("LONG1 of {n} bytes");
                }
                let mut v: i64 = 0;
                for (k, &x) in b.iter().enumerate() {
                    v |= (x as i64) << (8 * k);
                }
                if n > 0 && n < 8 && b[n - 1] & 0x80 != 0 {
                    v -= 1i64 << (8 * n);
                }
                stack.push(Val::Int(v));
            }
            b'G' => stack.push(Val::Float(f64::from_be_bytes(take(&mut i, 8)?.try_into()?))),
            b'c' => {
                // GLOBAL "module\nname\n"
                let read_line = |i: &mut usize| -> Result<String> {
                    let end = data[*i..]
                        .iter()
                        .position(|&c| c == b'\n')
                        .ok_or_else(|| anyhow!("GLOBAL without newline"))?;
                    let s = String::from_utf8(data[*i..*i + end].to_vec())?;
                    *i += end + 1;
                    Ok(s)
                };
                let module = read_line(&mut i)?;
                let name = read_line(&mut i)?;
                stack.push(Val::Global(module, name));
            }
            0x93 => {
                // STACK_GLOBAL
                let name = pop(&mut stack)?;
                let module = pop(&mut stack)?;
                match (module, name) {
                    (Val::Str(m), Val::Str(n)) => stack.push(Val::Global(m, n)),
                    other => bail!("STACK_GLOBAL of {other:?}"),
                }
            }
            b't' => {
                let items = pop_mark(&mut stack)?;
                stack.push(Val::Tuple(items));
            }
            0x85 => {
                let a = pop(&mut stack)?;
                stack.push(Val::Tuple(vec![a]));
            }
            0x86 => {
                let b = pop(&mut stack)?;
                let a = pop(&mut stack)?;
                stack.push(Val::Tuple(vec![a, b]));
            }
            0x87 => {
                let c = pop(&mut stack)?;
                let b = pop(&mut stack)?;
                let a = pop(&mut stack)?;
                stack.push(Val::Tuple(vec![a, b, c]));
            }
            b'Q' => {
                // BINPERSID: ('storage', torch.<T>Storage, key, location, numel)
                let pid = pop(&mut stack)?;
                let Val::Tuple(t) = pid else {
                    bail!("persistent id {pid:?}")
                };
                match t.as_slice() {
                    [Val::Str(kind), Val::Global(_, ty), Val::Str(key), _, _]
                        if kind == "storage" =>
                    {
                        stack.push(Val::Storage {
                            dtype: TorchDtype::from_storage(ty)?,
                            key: key.clone(),
                        })
                    }
                    other => bail!("unsupported persistent id {other:?}"),
                }
            }
            b'R' => {
                // REDUCE
                let args = pop(&mut stack)?;
                let f = pop(&mut stack)?;
                let v = match (&f, args) {
                    (Val::Global(m, n), Val::Tuple(a))
                        if m == "torch._utils" && n == "_rebuild_tensor_v2" =>
                    {
                        let Some(Val::Storage { dtype, key }) = a.first().cloned() else {
                            bail!("_rebuild_tensor_v2 without a storage");
                        };
                        Val::Tensor {
                            dtype,
                            key,
                            offset: as_usize(a.get(1).ok_or_else(|| anyhow!("no offset"))?)?,
                            shape: as_usizes(a.get(2).ok_or_else(|| anyhow!("no shape"))?)?,
                            stride: as_usizes(a.get(3).ok_or_else(|| anyhow!("no stride"))?)?,
                        }
                    }
                    (Val::Global(m, n), _) if m == "collections" && n == "OrderedDict" => {
                        Val::Dict(Vec::new())
                    }
                    _ => Val::Opaque,
                };
                stack.push(v);
            }
            b'b' => {
                // BUILD: state is ignored (dict subclasses carry no tensor there).
                pop(&mut stack)?;
            }
            b's' => {
                let v = pop(&mut stack)?;
                let k = pop(&mut stack)?;
                match stack.last_mut() {
                    Some(Val::Dict(d)) => d.push((k, v)),
                    _ => bail!("SETITEM on a non-dict"),
                }
            }
            b'u' => {
                let items = pop_mark(&mut stack)?;
                if items.len() % 2 != 0 {
                    bail!("SETITEMS with an odd item count");
                }
                match stack.last_mut() {
                    Some(Val::Dict(d)) => {
                        let mut it = items.into_iter();
                        while let (Some(k), Some(v)) = (it.next(), it.next()) {
                            d.push((k, v));
                        }
                    }
                    _ => bail!("SETITEMS on a non-dict"),
                }
            }
            b'a' => {
                let v = pop(&mut stack)?;
                match stack.last_mut() {
                    Some(Val::List(l)) => l.push(v),
                    _ => bail!("APPEND on a non-list"),
                }
            }
            b'e' => {
                let items = pop_mark(&mut stack)?;
                match stack.last_mut() {
                    Some(Val::List(l)) => l.extend(items),
                    _ => bail!("APPENDS on a non-list"),
                }
            }
            other => bail!("unsupported pickle opcode 0x{other:02x} at byte {}", i - 1),
        }
    }
}

/// Reads every tensor of a `torch.save` state dict, as f32.
pub fn read_state_dict(path: &Path) -> Result<Vec<(String, TorchTensor)>> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut zip =
        zip::ZipArchive::new(file).with_context(|| format!("{} is not a zip", path.display()))?;
    let pkl_name: String = zip
        .file_names()
        .filter_map(|n| n.ok().map(|c| c.into_owned()))
        .find(|n| n.ends_with("/data.pkl") || n == "data.pkl")
        .ok_or_else(|| anyhow!("{}: no data.pkl", path.display()))?;
    let prefix = pkl_name.trim_end_matches("data.pkl").to_string();
    if let Ok(mut f) = zip.by_name(&format!("{prefix}byteorder")) {
        let mut s = String::new();
        f.read_to_string(&mut s)?;
        if s.trim() != "little" {
            bail!("{}: byte order {s:?}", path.display());
        }
    }
    let mut pkl = Vec::new();
    zip.by_name(&pkl_name)?.read_to_end(&mut pkl)?;
    let root = unpickle(&pkl)?;
    let Val::Dict(entries) = root else {
        bail!("{}: the checkpoint is not a dict", path.display());
    };
    let mut storages: HashMap<String, Vec<u8>> = HashMap::new();
    let mut out = Vec::with_capacity(entries.len());
    for (k, v) in entries {
        let Val::Str(name) = k else {
            bail!("non-string state dict key {k:?}");
        };
        let Val::Tensor {
            dtype,
            key,
            offset,
            shape,
            stride,
        } = v
        else {
            bail!("{name}: not a tensor");
        };
        if !storages.contains_key(&key) {
            let mut buf = Vec::new();
            zip.by_name(&format!("{prefix}data/{key}"))
                .with_context(|| format!("{name}: storage {key}"))?
                .read_to_end(&mut buf)?;
            storages.insert(key.clone(), buf);
        }
        let raw = &storages[&key];
        let n: usize = shape.iter().product();
        let es = dtype.size();
        // Element index of every output position (row-major walk of `shape` with `stride`).
        let mut data = Vec::with_capacity(n);
        let mut idx = vec![0usize; shape.len()];
        for _ in 0..n {
            let e = offset + idx.iter().zip(&stride).map(|(i, s)| i * s).sum::<usize>();
            let b = raw
                .get(e * es..e * es + es)
                .ok_or_else(|| anyhow!("{name}: element {e} outside its storage"))?;
            data.push(match dtype {
                TorchDtype::F32 => f32::from_le_bytes(b.try_into()?),
                TorchDtype::F16 => f16_to_f32(u16::from_le_bytes(b.try_into()?)),
                TorchDtype::BF16 => {
                    f32::from_bits((u16::from_le_bytes(b.try_into()?) as u32) << 16)
                }
            });
            for d in (0..shape.len()).rev() {
                idx[d] += 1;
                if idx[d] < shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        out.push((name, TorchTensor { shape, data }));
    }
    Ok(out)
}

/// IEEE half to f32.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match (exp, frac) {
        (0, 0) => sign,
        (0, f) => {
            // Subnormal: normalize.
            let mut e = 127 - 15 + 1;
            let mut f = f;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((f & 0x3ff) << 13)
        }
        (0x1f, f) => sign | 0x7f80_0000 | (f << 13),
        (e, f) => sign | ((e + 127 - 15) << 23) | (f << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A pickle writer for the few opcodes `torch.save` uses.
    #[derive(Default)]
    struct P(Vec<u8>);

    impl P {
        fn op(&mut self, b: &[u8]) -> &mut Self {
            self.0.extend_from_slice(b);
            self
        }
        fn str(&mut self, s: &str) -> &mut Self {
            self.0.push(b'X');
            self.0.extend((s.len() as u32).to_le_bytes());
            self.0.extend(s.as_bytes());
            self
        }
        fn int(&mut self, v: u32) -> &mut Self {
            self.0.push(b'J');
            self.0.extend(v.to_le_bytes());
            self
        }
        fn global(&mut self, m: &str, n: &str) -> &mut Self {
            self.0.push(b'c');
            self.0.extend(format!("{m}\n{n}\n").as_bytes());
            self
        }
        fn ints(&mut self, v: &[u32]) -> &mut Self {
            self.op(b"(");
            for &x in v {
                self.int(x);
            }
            self.op(b"t")
        }
        /// `name: _rebuild_tensor_v2(storage(key, ty), offset, shape, stride, False, OrderedDict())`
        #[allow(clippy::too_many_arguments)]
        fn tensor(
            &mut self,
            name: &str,
            ty: &str,
            key: &str,
            numel: u32,
            offset: u32,
            shape: &[u32],
            stride: &[u32],
        ) -> &mut Self {
            self.str(name)
                .global("torch._utils", "_rebuild_tensor_v2")
                .op(b"((")
                .str("storage")
                .global("torch", ty)
                .str(key)
                .str("cpu")
                .int(numel)
                .op(b"tQ")
                .int(offset)
                .ints(shape)
                .ints(stride)
                .op(&[0x89])
                .global("collections", "OrderedDict")
                .op(b")R")
                .op(b"tR")
        }
    }

    fn write_zip(path: &Path, files: &[(&str, Vec<u8>)]) {
        let f = std::fs::File::create(path).unwrap();
        let mut z = zip::ZipWriter::new(f);
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        for (name, data) in files {
            z.start_file(*name, opts).unwrap();
            z.write_all(data).unwrap();
        }
        z.finish().unwrap();
    }

    #[test]
    fn reads_tensors_strides_and_dtypes() {
        // "a": [2, 3] f32 contiguous; "b": its transpose (a strided view of the same storage);
        // "c": a 2-element f16 tensor at offset 1 of a 3-element storage; "d": bf16.
        let a: Vec<f32> = (0..6).map(|i| i as f32 * 0.5 - 1.0).collect();
        let f16: Vec<u16> = vec![0x3c00, 0xc000, 0x3800]; // 1.0, -2.0, 0.5
        let bf16: Vec<u16> = vec![0x3f80, 0x4040]; // 1.0, 3.0
        let mut p = P::default();
        p.op(&[0x80, 2]).op(b"}q\x00(");
        p.tensor("a", "FloatStorage", "0", 6, 0, &[2, 3], &[3, 1]);
        p.tensor("b", "FloatStorage", "0", 6, 0, &[3, 2], &[1, 3]);
        p.tensor("c", "HalfStorage", "1", 3, 1, &[2], &[1]);
        p.tensor("d", "BFloat16Storage", "2", 2, 0, &[1, 2], &[2, 1]);
        p.op(b"u.");
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("pytorch_model.bin");
        write_zip(
            &path,
            &[
                ("archive/data.pkl", p.0.clone()),
                ("archive/byteorder", b"little".to_vec()),
                (
                    "archive/data/0",
                    a.iter().flat_map(|x| x.to_le_bytes()).collect(),
                ),
                (
                    "archive/data/1",
                    f16.iter().flat_map(|x| x.to_le_bytes()).collect(),
                ),
                (
                    "archive/data/2",
                    bf16.iter().flat_map(|x| x.to_le_bytes()).collect(),
                ),
            ],
        );
        let sd = read_state_dict(&path).unwrap();
        let names: Vec<&str> = sd.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["a", "b", "c", "d"]);
        assert_eq!(sd[0].1.shape, [2, 3]);
        assert_eq!(sd[0].1.data, a);
        assert_eq!(sd[1].1.shape, [3, 2]);
        assert_eq!(sd[1].1.data, [-1.0, 0.5, -0.5, 1.0, 0.0, 1.5]);
        assert_eq!(sd[2].1.data, [-2.0, 0.5]);
        assert_eq!(sd[3].1.data, [1.0, 3.0]);
    }

    #[test]
    fn rejects_code_and_bad_storages() {
        // An unknown opcode (here INST, `i`) is an error, never executed.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("x.bin");
        write_zip(&path, &[("x/data.pkl", b"\x80\x02ios\nsystem\n.".to_vec())]);
        assert!(read_state_dict(&path).is_err());
        // A tensor reaching past its storage.
        let mut p = P::default();
        p.op(&[0x80, 2]).op(b"}(");
        p.tensor("a", "FloatStorage", "0", 2, 1, &[2], &[1]);
        p.op(b"u.");
        write_zip(
            &path,
            &[("x/data.pkl", p.0.clone()), ("x/data/0", vec![0u8; 8])],
        );
        assert!(read_state_dict(&path).is_err());
    }

    #[test]
    fn half_precision_conversion() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);
        assert!(f16_to_f32(0x7c00).is_infinite());
    }
}
