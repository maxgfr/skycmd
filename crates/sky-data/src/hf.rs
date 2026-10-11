//! Hugging Face Hub access over plain HTTPS (public, ungated datasets only) and
//! parquet reading, including partial reads of huge remote shards via HTTP range
//! requests into a sparse local cache file.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::sleep;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::schema::types::Type;
use serde_json::Value;

const HUB: &str = "https://huggingface.co";
const RETRIES: u32 = 4;

/// URL of a file in a dataset repository at a given revision.
pub fn resolve_url(repo: &str, rev: &str, file: &str) -> String {
    let rev = rev.replace('/', "%2F");
    format!("{HUB}/datasets/{repo}/resolve/{rev}/{file}")
}

/// Hub client with an on-disk cache.
pub struct Hf {
    agent: ureq::Agent,
    pub cache: PathBuf,
}

impl Hf {
    pub fn new(cache: impl Into<PathBuf>) -> Result<Self> {
        let cache = cache.into();
        fs::create_dir_all(&cache)?;
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(600)))
            .user_agent("skycmd-sky-data/0.1")
            .build()
            .into();
        Ok(Self { agent, cache })
    }

    fn retry<T>(&self, what: &str, mut f: impl FnMut() -> Result<T>) -> Result<T> {
        let mut last = None;
        for attempt in 0..RETRIES {
            match f() {
                Ok(v) => return Ok(v),
                Err(e) => {
                    eprintln!("  retry {}/{RETRIES} {what}: {e:#}", attempt + 1);
                    last = Some(e);
                    sleep(Duration::from_millis(500 << attempt));
                }
            }
        }
        Err(last.unwrap_or_else(|| anyhow!("no attempt made")))
    }

    /// GET a JSON document.
    pub fn get_json(&self, url: &str) -> Result<Value> {
        self.retry(url, || {
            let mut resp = self.agent.get(url).call()?;
            Ok(resp.body_mut().with_config().limit(64 << 20).read_json()?)
        })
    }

    /// GET a whole body into memory (bounded by `limit` bytes).
    pub fn get_bytes(&self, url: &str, limit: u64) -> Result<Vec<u8>> {
        self.retry(url, || {
            let mut resp = self.agent.get(url).call()?;
            Ok(resp.body_mut().with_config().limit(limit).read_to_vec()?)
        })
    }

    /// Downloads `url` to `cache/rel` unless it is already cached. Refuses files larger than `max_bytes`.
    pub fn download(&self, url: &str, rel: &str, max_bytes: u64) -> Result<PathBuf> {
        let dest = self.cache.join(rel);
        if dest.exists() && !ranges_path(&dest).exists() {
            return Ok(dest);
        }
        if let Some(dir) = dest.parent() {
            fs::create_dir_all(dir)?;
        }
        let tmp = dest.with_extension("part");
        self.retry(url, || {
            let resp = self.agent.get(url).call()?;
            let mut reader = resp.into_body().into_with_config().limit(max_bytes).reader();
            let mut out = File::create(&tmp)?;
            std::io::copy(&mut reader, &mut out).with_context(|| format!("download {url}"))?;
            out.flush()?;
            Ok(())
        })?;
        fs::rename(&tmp, &dest)?;
        Ok(dest)
    }

    /// Parquet shard URLs for `config`/`split` from the dataset-viewer conversion
    /// (`/api/datasets/<repo>/parquet/<config>/<split>`), which also covers repos
    /// without native parquet files (`refs/convert/parquet`).
    pub fn parquet_urls(&self, repo: &str, config: &str, split: &str) -> Result<Vec<String>> {
        let url = format!("{HUB}/api/datasets/{repo}/parquet/{config}/{split}");
        let v = self.get_json(&url)?;
        let urls: Vec<String> = v
            .as_array()
            .ok_or_else(|| anyhow!("unexpected parquet listing for {repo}: {v}"))?
            .iter()
            .filter_map(|u| u.as_str().map(str::to_string))
            .collect();
        if urls.is_empty() {
            bail!("no parquet shards for {repo} {config}/{split}");
        }
        Ok(urls)
    }

    /// Downloads every parquet shard of a split (small datasets) and returns the local paths.
    pub fn parquet_split(&self, repo: &str, config: &str, split: &str, max_bytes: u64) -> Result<Vec<PathBuf>> {
        let urls = self.parquet_urls(repo, config, split)?;
        urls.iter()
            .enumerate()
            .map(|(i, u)| {
                let rel = format!("{}/{config}/{split}-{i:05}.parquet", repo.replace('/', "__"));
                self.download(u, &rel, max_bytes)
            })
            .collect()
    }

    /// Opens a remote parquet file for partial reading. Only the footer is fetched now.
    pub fn remote_parquet(&self, url: &str, rel: &str) -> Result<RemoteParquet<'_>> {
        RemoteParquet::open(self, url, rel)
    }

    /// Fetches `[start, start+len)` with a range request.
    fn get_range(&self, url: &str, start: u64, len: u64) -> Result<(Vec<u8>, Option<u64>)> {
        self.retry(url, || {
            let resp = self
                .agent
                .get(url)
                .header("Range", &format!("bytes={}-{}", start, start + len - 1))
                .call()?;
            if resp.status().as_u16() != 206 {
                bail!("range request not honoured (status {})", resp.status());
            }
            let total = resp
                .headers()
                .get("content-range")
                .and_then(|h| h.to_str().ok())
                .and_then(|s| s.rsplit('/').next())
                .and_then(|s| s.parse::<u64>().ok());
            let mut buf = Vec::with_capacity(len as usize);
            resp.into_body()
                .into_with_config()
                .limit(len + 1)
                .reader()
                .read_to_end(&mut buf)?;
            if buf.len() as u64 != len {
                bail!("short range read: {} of {len} bytes", buf.len());
            }
            Ok((buf, total))
        })
    }
}

fn ranges_path(p: &Path) -> PathBuf {
    let mut s = p.as_os_str().to_owned();
    s.push(".ranges");
    PathBuf::from(s)
}

/// A remote parquet file mirrored into a sparse local file: only the footer and the
/// column chunks that are actually read get downloaded. Fetched ranges are recorded
/// in a `.ranges` sidecar so later runs reuse them.
pub struct RemoteParquet<'a> {
    hf: &'a Hf,
    url: String,
    path: PathBuf,
    fetched: Vec<(u64, u64)>,
    reader: SerializedFileReader<File>,
}

impl<'a> RemoteParquet<'a> {
    fn open(hf: &'a Hf, url: &str, rel: &str) -> Result<Self> {
        let path = hf.cache.join(rel);
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        let side = ranges_path(&path);
        let complete = path.exists() && !side.exists();
        let mut fetched: Vec<(u64, u64)> = if side.exists() {
            serde_json::from_slice(&fs::read(&side)?).unwrap_or_default()
        } else {
            Vec::new()
        };
        if !path.exists() || (side.exists() && fetched.is_empty()) {
            // Footer: last 8 bytes hold the metadata length and the PAR1 magic.
            let (tail8, total) = hf.get_range(url, 0, 8).and_then(|(head, total)| {
                if &head[..4] != b"PAR1" {
                    bail!("not a parquet file: {url}");
                }
                let total = total.ok_or_else(|| anyhow!("no content-range total for {url}"))?;
                Ok((hf.get_range(url, total - 8, 8)?.0, total))
            })?;
            if &tail8[4..] != b"PAR1" {
                bail!("bad parquet footer magic: {url}");
            }
            let meta_len = u32::from_le_bytes(tail8[..4].try_into().unwrap()) as u64;
            let start = total - 8 - meta_len;
            let (footer, _) = hf.get_range(url, start, meta_len + 8)?;
            let mut f = File::create(&path)?;
            f.set_len(total)?;
            f.write_all(b"PAR1")?;
            f.seek(SeekFrom::Start(start))?;
            f.write_all(&footer)?;
            fetched = vec![(0, 4), (start, total)];
            fs::write(&side, serde_json::to_vec(&fetched)?)?;
        } else if complete {
            fetched = vec![(0, fs::metadata(&path)?.len())];
        }
        let reader = SerializedFileReader::new(File::open(&path)?)
            .with_context(|| format!("parse parquet footer {path:?}"))?;
        Ok(Self {
            hf,
            url: url.to_string(),
            path,
            fetched,
            reader,
        })
    }

    pub fn num_row_groups(&self) -> usize {
        self.reader.num_row_groups()
    }

    pub fn row_group_rows(&self, rg: usize) -> i64 {
        self.reader.metadata().row_group(rg).num_rows()
    }

    fn has(&self, start: u64, end: u64) -> bool {
        self.fetched.iter().any(|&(a, b)| a <= start && end <= b)
    }

    /// Makes sure the chunks of `columns` in row group `rg` are present locally.
    pub fn ensure(&mut self, rg: usize, columns: &[&str]) -> Result<u64> {
        let meta = self.reader.metadata().row_group(rg).clone();
        let mut fetched_bytes = 0;
        for col in meta.columns() {
            let name = col.column_path().parts().first().cloned().unwrap_or_default();
            if !columns.contains(&name.as_str()) {
                continue;
            }
            let (start, len) = col.byte_range();
            if len == 0 || self.has(start, start + len) {
                continue;
            }
            let (buf, _) = self.hf.get_range(&self.url, start, len)?;
            let mut f = OpenOptions::new().write(true).open(&self.path)?;
            f.seek(SeekFrom::Start(start))?;
            f.write_all(&buf)?;
            f.flush()?;
            self.fetched.push((start, start + len));
            fs::write(ranges_path(&self.path), serde_json::to_vec(&self.fetched)?)?;
            fetched_bytes += len;
        }
        Ok(fetched_bytes)
    }

    /// Reads the given top-level columns of one row group as JSON rows.
    pub fn rows(&mut self, rg: usize, columns: &[&str]) -> Result<Vec<Value>> {
        self.ensure(rg, columns)?;
        // Re-open so the reader sees the freshly written bytes.
        self.reader = SerializedFileReader::new(File::open(&self.path)?)?;
        rows_of_group(&self.reader, rg, columns)
    }
}

fn projection(reader: &SerializedFileReader<File>, columns: &[&str]) -> Result<Option<Type>> {
    if columns.is_empty() {
        return Ok(None);
    }
    let root = reader.metadata().file_metadata().schema_descr().root_schema();
    let fields: Vec<Arc<Type>> = root
        .get_fields()
        .iter()
        .filter(|f| columns.contains(&f.name()))
        .cloned()
        .collect();
    if fields.len() != columns.len() {
        let have: Vec<&str> = root.get_fields().iter().map(|f| f.name()).collect();
        bail!("missing columns {columns:?}; file has {have:?}");
    }
    Ok(Some(Type::group_type_builder("schema").with_fields(fields).build()?))
}

fn rows_of_group(reader: &SerializedFileReader<File>, rg: usize, columns: &[&str]) -> Result<Vec<Value>> {
    let proj = projection(reader, columns)?;
    let group = reader.get_row_group(rg)?;
    let mut out = Vec::with_capacity(group.metadata().num_rows() as usize);
    for row in group.get_row_iter(proj)? {
        out.push(row?.to_json_value());
    }
    Ok(out)
}

/// Reads up to `max_rows` rows (projected to `columns`, all when empty) of a local parquet file.
pub fn read_parquet(path: &Path, columns: &[&str], max_rows: usize) -> Result<Vec<Value>> {
    let reader = SerializedFileReader::new(File::open(path).with_context(|| format!("open {path:?}"))?)?;
    let mut out = Vec::new();
    for rg in 0..reader.num_row_groups() {
        if out.len() >= max_rows {
            break;
        }
        let mut rows = rows_of_group(&reader, rg, columns)?;
        rows.truncate(max_rows - out.len());
        out.extend(rows);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_url_escapes_refs() {
        assert_eq!(
            resolve_url("a/b", "refs/convert/parquet", "x/y.parquet"),
            "https://huggingface.co/datasets/a/b/resolve/refs%2Fconvert%2Fparquet/x/y.parquet"
        );
    }

    #[test]
    fn sidecar_name() {
        assert_eq!(ranges_path(Path::new("/tmp/a.parquet")), PathBuf::from("/tmp/a.parquet.ranges"));
    }
}
