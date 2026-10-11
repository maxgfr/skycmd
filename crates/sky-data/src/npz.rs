//! Minimal `.npz` / `.npy` reader for 2-D little-endian `uint16` arrays
//! (enough for token-id matrices such as `pngwn/typed-decisions`).

use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};

/// A row-major 2-D `u16` matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct U16Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<u16>,
}

impl U16Matrix {
    pub fn row(&self, i: usize) -> &[u16] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }
}

/// Reads array `name` (without the `.npy` suffix) from an `.npz` archive.
pub fn read_npz_u16(path: &Path, name: &str) -> Result<U16Matrix> {
    let f = File::open(path).with_context(|| format!("open {path:?}"))?;
    let mut zip = zip::ZipArchive::new(f)?;
    let mut entry = zip
        .by_name(&format!("{name}.npy"))
        .with_context(|| format!("{name}.npy not in {path:?}"))?;
    let mut buf = Vec::with_capacity(entry.size() as usize);
    entry.read_to_end(&mut buf)?;
    parse_npy_u16(&buf)
}

/// Parses an in-memory `.npy` file holding a C-ordered 2-D `<u2` array.
pub fn parse_npy_u16(buf: &[u8]) -> Result<U16Matrix> {
    if buf.len() < 10 || &buf[..6] != b"\x93NUMPY" {
        bail!("not an npy file");
    }
    let major = buf[6];
    let (hlen, off) = match major {
        1 => (u16::from_le_bytes([buf[8], buf[9]]) as usize, 10),
        2 | 3 => {
            if buf.len() < 12 {
                bail!("truncated npy header");
            }
            (u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize, 12)
        }
        v => bail!("unsupported npy version {v}"),
    };
    let header = std::str::from_utf8(buf.get(off..off + hlen).ok_or_else(|| anyhow!("truncated npy header"))?)?;
    let descr = dict_value(header, "descr").ok_or_else(|| anyhow!("no descr in {header}"))?;
    if !matches!(descr.trim_matches(|c| c == '\'' || c == '"'), "<u2" | "|u2" | "u2") {
        bail!("unsupported dtype {descr}");
    }
    if dict_value(header, "fortran_order").is_some_and(|v| v.starts_with("True")) {
        bail!("fortran-ordered arrays are not supported");
    }
    let shape = dict_value(header, "shape").ok_or_else(|| anyhow!("no shape in {header}"))?;
    let dims: Vec<usize> = shape
        .trim_start_matches('(')
        .split(')')
        .next()
        .unwrap_or("")
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>())
        .collect::<std::result::Result<_, _>>()?;
    let (rows, cols) = match dims.as_slice() {
        [r, c] => (*r, *c),
        [n] => (1, *n),
        _ => bail!("expected a 2-D array, got shape {dims:?}"),
    };
    let body = &buf[off + hlen..];
    if body.len() < rows * cols * 2 {
        bail!("npy body too short");
    }
    let data = body[..rows * cols * 2]
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    Ok(U16Matrix { rows, cols, data })
}

/// Extracts the raw value of `'key':` in a numpy header dict (a tuple or a scalar token).
fn dict_value<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("'{key}':");
    let i = header.find(&pat)? + pat.len();
    let rest = header[i..].trim_start();
    let end = if rest.starts_with('(') {
        rest.find(')')? + 1
    } else {
        rest.find([',', '}']).unwrap_or(rest.len())
    };
    Some(rest[..end].trim())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn npy(shape: &str, vals: &[u16]) -> Vec<u8> {
        let mut header = format!("{{'descr': '<u2', 'fortran_order': False, 'shape': {shape}, }}");
        while (10 + header.len() + 1) % 64 != 0 {
            header.push(' ');
        }
        header.push('\n');
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend((header.len() as u16).to_le_bytes());
        out.extend(header.as_bytes());
        for v in vals {
            out.extend(v.to_le_bytes());
        }
        out
    }

    #[test]
    fn parses_2d() {
        let m = parse_npy_u16(&npy("(2, 3)", &[1, 2, 3, 4, 5, 50256])).unwrap();
        assert_eq!((m.rows, m.cols), (2, 3));
        assert_eq!(m.row(1), &[4, 5, 50256]);
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse_npy_u16(b"hello world").is_err());
    }
}
