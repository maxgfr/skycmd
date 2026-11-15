//! Tool clusters and the split by cluster (section 4.2), and the nearest-tool lists of the hard
//! slates (4.4), from the embedding files `scripts/route-embed.py` writes (`<emb>.npy`: one
//! row of float32 per line of the texts file, in its order).

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use anyhow::{bail, Context, Result};

/// A row-major float32 matrix, rows L2-normalised.
#[derive(Clone, Debug, Default)]
pub struct Matrix {
    pub rows: usize,
    pub dim: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    /// From rows; each is L2-normalised (a zero row stays zero).
    pub fn from_rows(rows: &[Vec<f32>]) -> Self {
        let dim = rows.first().map_or(0, Vec::len);
        let mut m = Self {
            rows: rows.len(),
            dim,
            data: rows.concat(),
        };
        m.normalise();
        m
    }

    fn normalise(&mut self) {
        for i in 0..self.rows {
            let r = &mut self.data[i * self.dim..(i + 1) * self.dim];
            let n = r.iter().map(|x| x * x).sum::<f32>().sqrt();
            if n > 0.0 {
                r.iter_mut().for_each(|x| *x /= n);
            }
        }
    }

    /// A version 1 or 2 `.npy` file of little-endian float32, C order, two dimensions.
    pub fn read_npy(path: &Path) -> Result<Self> {
        let buf = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        if buf.len() < 10 || &buf[..6] != b"\x93NUMPY" {
            bail!("{}: not a .npy file", path.display());
        }
        let (hlen, at) = match buf[6] {
            1 => (u16::from_le_bytes([buf[8], buf[9]]) as usize, 10),
            _ => (
                u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize,
                12,
            ),
        };
        let header = std::str::from_utf8(&buf[at..at + hlen])?;
        if !header.contains("'descr': '<f4'") || header.contains("'fortran_order': True") {
            bail!(
                "{}: not a C-order float32 matrix ({header})",
                path.display()
            );
        }
        let shape = header
            .split("'shape': (")
            .nth(1)
            .and_then(|s| s.split(')').next())
            .ok_or_else(|| anyhow::anyhow!("{}: no shape", path.display()))?;
        let dims: Vec<usize> = shape
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse())
            .collect::<Result<_, _>>()?;
        if dims.len() != 2 {
            bail!(
                "{}: shape {dims:?}, two dimensions expected",
                path.display()
            );
        }
        let body = &buf[at + hlen..];
        if body.len() != dims[0] * dims[1] * 4 {
            bail!(
                "{}: {} bytes for shape {dims:?}",
                path.display(),
                body.len()
            );
        }
        let data = body
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let mut m = Self {
            rows: dims[0],
            dim: dims[1],
            data,
        };
        m.normalise();
        Ok(m)
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The pairs `(i, j)`, `i < j`, of `rows` whose cosine is at least `threshold`, in order. The
/// work is shared between threads by row; the result does not depend on their number.
pub fn edges(m: &Matrix, rows: &[usize], threshold: f32) -> Vec<(usize, usize)> {
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(16);
    let chunks: Vec<Vec<(usize, usize)>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                s.spawn(move || {
                    let mut out = Vec::new();
                    for a in (t..rows.len()).step_by(threads) {
                        let ra = m.row(rows[a]);
                        for &rb in &rows[a + 1..] {
                            if dot(ra, m.row(rb)) >= threshold {
                                out.push((rows[a], rb));
                            }
                        }
                    }
                    out
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut all: Vec<(usize, usize)> = chunks.into_iter().flatten().collect();
    all.sort_unstable();
    all
}

/// Union-find with path halving; the root of a set is its smallest member.
pub struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    pub fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    pub fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
            self.parent[hi] = lo;
        }
    }
}

/// 4.2, item 2: two tools are one cluster if their normalised names are equal or their cosine is
/// at least `threshold`. `names[i]` is the normalised name of row `rows[i]`. Returns the cluster
/// of each of `rows` (the index of its first member in `rows`).
pub fn clusters(m: &Matrix, rows: &[usize], names: &[String], threshold: f32) -> Vec<usize> {
    let mut uf = UnionFind::new(rows.len());
    let mut first: HashMap<&str, usize> = HashMap::new();
    for (i, n) in names.iter().enumerate() {
        match first.get(n.as_str()) {
            Some(&j) => uf.union(i, j),
            None => {
                first.insert(n, i);
            }
        }
    }
    let at: HashMap<usize, usize> = rows.iter().enumerate().map(|(i, &r)| (r, i)).collect();
    for (a, b) in edges(m, rows, threshold) {
        uf.union(at[&a], at[&b]);
    }
    (0..rows.len()).map(|i| uf.find(i)).collect()
}

/// 4.2, item 3: the split of a cluster from the first byte of sha256(smallest normalised name +
/// `1234`), read as a fraction of 256: below 0.85 train, below 0.90 val, else test.
pub fn split_of_key(key: &str) -> &'static str {
    let b = super::salted(key)[0] as f64 / 256.0;
    if b < 0.85 {
        "train"
    } else if b < 0.90 {
        "val"
    } else {
        "test"
    }
}

/// The `k` rows of `pool` nearest to row `q` by cosine, those `skip` refuses left out; ties go
/// to the lower row.
pub fn nearest(
    m: &Matrix,
    q: usize,
    pool: &[usize],
    k: usize,
    skip: impl Fn(usize) -> bool,
) -> Vec<usize> {
    let rq = m.row(q);
    let mut best: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
    for &p in pool {
        if p == q || skip(p) {
            continue;
        }
        let s = dot(rq, m.row(p));
        if best.len() == k && !better((s, p), best[k - 1]) {
            continue;
        }
        let at = best.partition_point(|&e| better(e, (s, p)));
        best.insert(at, (s, p));
        best.truncate(k);
    }
    best.into_iter().map(|(_, p)| p).collect()
}

fn better(a: (f32, usize), b: (f32, usize)) -> bool {
    a.0 > b.0 || (a.0 == b.0 && a.1 < b.1)
}

/// [`nearest`] for many queries at once, on threads; `skip(q, p)`.
pub fn nearest_all(
    m: &Matrix,
    queries: &[usize],
    pool: &[usize],
    k: usize,
    skip: &(dyn Fn(usize, usize) -> bool + Sync),
) -> BTreeMap<usize, Vec<usize>> {
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(16);
    let parts: Vec<Vec<(usize, Vec<usize>)>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                s.spawn(move || {
                    queries
                        .iter()
                        .skip(t)
                        .step_by(threads)
                        .map(|&q| (q, nearest(m, q, pool, k, |p| skip(q, p))))
                        .collect()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    parts.into_iter().flatten().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m() -> Matrix {
        Matrix::from_rows(&[
            vec![1.0, 0.0, 0.0],
            vec![0.99, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 0.05, 1.0],
        ])
    }

    #[test]
    fn npy_round_trip() {
        let dir = std::env::temp_dir().join(format!("sky-route-npy-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }";
        let mut h = header.to_string();
        while !(10 + h.len() + 1).is_multiple_of(64) {
            h.push(' ');
        }
        h.push('\n');
        let mut buf = b"\x93NUMPY\x01\x00".to_vec();
        buf.extend_from_slice(&(h.len() as u16).to_le_bytes());
        buf.extend_from_slice(h.as_bytes());
        for x in [3.0f32, 4.0, 0.0, 0.0, 0.0, 2.0] {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        let p = dir.join("a.npy");
        std::fs::write(&p, buf).unwrap();
        let mm = Matrix::read_npy(&p).unwrap();
        assert_eq!((mm.rows, mm.dim), (2, 3));
        assert!((mm.row(0)[0] - 0.6).abs() < 1e-6 && (mm.row(1)[2] - 1.0).abs() < 1e-6);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn clusters_by_name_or_cosine() {
        let mm = m();
        let rows = vec![0, 1, 2, 3, 4];
        let names: Vec<String> = ["a", "b", "c", "c2", "d"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let e = edges(&mm, &rows, 0.95);
        assert_eq!(e, vec![(0, 1), (3, 4)]);
        let c = clusters(&mm, &rows, &names, 0.95);
        assert_eq!(c, vec![0, 0, 2, 3, 3]);
        // equal names join whatever the cosine
        let names: Vec<String> = ["a", "b", "a", "x", "y"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(clusters(&mm, &rows, &names, 0.999), vec![0, 1, 0, 3, 4]);
    }

    #[test]
    fn split_shares() {
        let mut n = BTreeMap::new();
        for i in 0..20_000 {
            *n.entry(split_of_key(&format!("tool{i}"))).or_insert(0usize) += 1;
        }
        let share = |k: &str| n[k] as f64 / 20_000.0;
        assert!((share("train") - 0.85).abs() < 0.02, "{n:?}");
        assert!((share("val") - 0.05).abs() < 0.01, "{n:?}");
        assert!((share("test") - 0.10).abs() < 0.015, "{n:?}");
        assert_eq!(split_of_key("tool1"), split_of_key("tool1"));
    }

    #[test]
    fn nearest_rows() {
        let mm = m();
        let pool = [0, 1, 2, 3, 4];
        assert_eq!(nearest(&mm, 0, &pool, 2, |_| false), vec![1, 2]);
        assert_eq!(nearest(&mm, 0, &pool, 2, |p| p == 1), vec![2, 3]);
        let all = nearest_all(&mm, &[0, 3], &pool, 1, &|_, _| false);
        assert_eq!(all[&0], vec![1]);
        assert_eq!(all[&3], vec![4]);
    }
}
