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

/// Kind of repository: a Hub model or dataset, or a GitHub repository (evaluation code and
/// stimuli published there, e.g. the 2609.33209 coherence battery).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepoKind {
    Model,
    Dataset,
    GitHub,
}

impl RepoKind {
    fn api(self) -> &'static str {
        match self {
            RepoKind::Model => "models",
            RepoKind::Dataset => "datasets",
            RepoKind::GitHub => "github",
        }
    }
}

/// URL of a file in a repository at a given revision.
pub fn resolve_repo_url(kind: RepoKind, repo: &str, rev: &str, file: &str) -> String {
    match kind {
        RepoKind::GitHub => format!("https://raw.githubusercontent.com/{repo}/{rev}/{file}"),
        RepoKind::Model => format!("{HUB}/{repo}/resolve/{}/{file}", rev.replace('/', "%2F")),
        RepoKind::Dataset => format!(
            "{HUB}/datasets/{repo}/resolve/{}/{file}",
            rev.replace('/', "%2F")
        ),
    }
}

/// One file of a repository listing, with the hash the Hub publishes for it: the sha256 of
/// the content for LFS files, the git blob sha1 otherwise.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepoFile {
    pub path: String,
    pub size: u64,
    pub sha256: Option<String>,
    pub git_sha1: Option<String>,
}

/// Hub client with an on-disk cache.
pub struct Hf {
    agent: ureq::Agent,
    /// `HF_TOKEN`, sent as a bearer token to huggingface.co only (gated repos).
    token: Option<String>,
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
        let token = std::env::var("HF_TOKEN")
            .ok()
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty());
        Ok(Self {
            agent,
            token,
            cache,
        })
    }

    /// A GET request, with the bearer token when the URL is on the Hub.
    fn get(&self, url: &str) -> ureq::RequestBuilder<ureq::typestate::WithoutBody> {
        let req = self.agent.get(url);
        match &self.token {
            Some(t) if url.starts_with(HUB) => req.header("Authorization", &format!("Bearer {t}")),
            _ => req,
        }
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
            let mut resp = self.get(url).call()?;
            Ok(resp.body_mut().with_config().limit(64 << 20).read_json()?)
        })
    }

    /// GET a whole body into memory (bounded by `limit` bytes).
    pub fn get_bytes(&self, url: &str, limit: u64) -> Result<Vec<u8>> {
        self.retry(url, || {
            let mut resp = self.get(url).call()?;
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
            let resp = self.get(url).call()?;
            let mut reader = resp
                .into_body()
                .into_with_config()
                .limit(max_bytes)
                .reader();
            let mut out = File::create(&tmp)?;
            std::io::copy(&mut reader, &mut out).with_context(|| format!("download {url}"))?;
            out.flush()?;
            Ok(())
        })?;
        fs::rename(&tmp, &dest)?;
        Ok(dest)
    }

    /// Every file of a repository at `rev` (recursive listing), with its published hash.
    pub fn repo_files(&self, kind: RepoKind, repo: &str, rev: &str) -> Result<Vec<RepoFile>> {
        if kind == RepoKind::GitHub {
            return self.github_files(repo, rev);
        }
        let mut url = Some(format!(
            "{HUB}/api/{}/{repo}/tree/{}?recursive=true",
            kind.api(),
            rev.replace('/', "%2F")
        ));
        // The listing is paginated (1000 entries per page, `Link: <next>; rel="next"`).
        let mut arr = Vec::new();
        while let Some(u) = url.take() {
            let (v, next) = self.retry(&u, || {
                let mut resp = self.get(&u).call()?;
                let next = resp
                    .headers()
                    .get("link")
                    .and_then(|h| h.to_str().ok())
                    .and_then(next_link);
                let v: Value = resp.body_mut().with_config().limit(64 << 20).read_json()?;
                Ok((v, next))
            })?;
            match v {
                Value::Array(a) => arr.extend(a),
                v => bail!("unexpected tree listing for {repo}: {v}"),
            }
            url = next;
        }
        Ok(arr
            .iter()
            .filter(|e| e["type"].as_str() == Some("file"))
            .filter_map(|e| {
                let lfs = e["lfs"]["oid"].as_str().map(str::to_string);
                Some(RepoFile {
                    path: e["path"].as_str()?.to_string(),
                    size: e["size"].as_u64().unwrap_or(0),
                    git_sha1: if lfs.is_none() {
                        e["oid"].as_str().map(str::to_string)
                    } else {
                        None
                    },
                    sha256: lfs,
                })
            })
            .collect())
    }

    /// GitHub tree listing: every blob with its blob sha1.
    fn github_files(&self, repo: &str, commit: &str) -> Result<Vec<RepoFile>> {
        let url = format!("https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1");
        let v = self.get_json(&url)?;
        if v["truncated"].as_bool() == Some(true) {
            bail!("{repo}: truncated tree listing");
        }
        Ok(v["tree"]
            .as_array()
            .ok_or_else(|| anyhow!("unexpected tree listing for {repo}: {v}"))?
            .iter()
            .filter(|e| e["type"].as_str() == Some("blob"))
            .filter_map(|e| {
                Some(RepoFile {
                    path: e["path"].as_str()?.to_string(),
                    size: e["size"].as_u64()?,
                    sha256: None,
                    git_sha1: e["sha"].as_str().map(str::to_string),
                })
            })
            .collect())
    }

    /// The commit sha a revision (branch, tag or sha) points to.
    pub fn repo_commit(&self, kind: RepoKind, repo: &str, rev: &str) -> Result<String> {
        if kind == RepoKind::GitHub {
            let v = self.get_json(&format!(
                "https://api.github.com/repos/{repo}/commits/{rev}"
            ))?;
            return v["sha"]
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| anyhow!("{repo}@{rev}: no commit sha"));
        }
        let url = format!(
            "{HUB}/api/{}/{repo}/revision/{}",
            kind.api(),
            rev.replace('/', "%2F")
        );
        let v = self.get_json(&url)?;
        v["sha"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| anyhow!("{repo}@{rev}: no commit sha in {url}"))
    }

    /// Downloads `file` of a repository to `dest` and checks it against the published hash.
    /// An existing `dest` with the right hash is kept (returns false); a wrong one is replaced.
    pub fn fetch_verified(
        &self,
        kind: RepoKind,
        repo: &str,
        rev: &str,
        file: &RepoFile,
        dest: &Path,
    ) -> Result<bool> {
        if dest.exists() && verify_file(dest, file).is_ok() {
            return Ok(false);
        }
        if let Some(dir) = dest.parent() {
            fs::create_dir_all(dir)?;
        }
        let url = resolve_repo_url(kind, repo, rev, &file.path);
        let tmp = dest.with_extension("part");
        self.retry(&url, || {
            let resp = self.get(&url).call()?;
            let mut reader = resp
                .into_body()
                .into_with_config()
                .limit(file.size + (1 << 20))
                .reader();
            let mut out = File::create(&tmp)?;
            std::io::copy(&mut reader, &mut out).with_context(|| format!("download {url}"))?;
            out.flush()?;
            verify_file(&tmp, file)
        })?;
        fs::rename(&tmp, dest)?;
        Ok(true)
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
    pub fn parquet_split(
        &self,
        repo: &str,
        config: &str,
        split: &str,
        max_bytes: u64,
    ) -> Result<Vec<PathBuf>> {
        let urls = self.parquet_urls(repo, config, split)?;
        urls.iter()
            .enumerate()
            .map(|(i, u)| {
                let rel = format!(
                    "{}/{config}/{split}-{i:05}.parquet",
                    repo.replace('/', "__")
                );
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
            let name = col
                .column_path()
                .parts()
                .first()
                .cloned()
                .unwrap_or_default();
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
    let root = reader
        .metadata()
        .file_metadata()
        .schema_descr()
        .root_schema();
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
    Ok(Some(
        Type::group_type_builder("schema")
            .with_fields(fields)
            .build()?,
    ))
}

fn rows_of_group(
    reader: &SerializedFileReader<File>,
    rg: usize,
    columns: &[&str],
) -> Result<Vec<Value>> {
    let proj = projection(reader, columns)?;
    let group = reader.get_row_group(rg)?;
    let mut out = Vec::with_capacity(group.metadata().num_rows() as usize);
    for row in group.get_row_iter(proj)? {
        out.push(row?.to_json_value());
    }
    Ok(out)
}

/// Top-level column names of a local parquet file.
pub fn parquet_columns(path: &Path) -> Result<Vec<String>> {
    let reader =
        SerializedFileReader::new(File::open(path).with_context(|| format!("open {path:?}"))?)?;
    let root = reader
        .metadata()
        .file_metadata()
        .schema_descr()
        .root_schema()
        .clone();
    Ok(root
        .get_fields()
        .iter()
        .map(|f| f.name().to_string())
        .collect())
}

/// Streams the rows (projected to `columns`) of a local parquet file through `f`, one row
/// group at a time, so huge files never sit in memory as JSON.
pub fn for_each_parquet_row(
    path: &Path,
    columns: &[&str],
    mut f: impl FnMut(Value) -> Result<()>,
) -> Result<()> {
    let reader =
        SerializedFileReader::new(File::open(path).with_context(|| format!("open {path:?}"))?)?;
    let proj = projection(&reader, columns)?;
    for rg in 0..reader.num_row_groups() {
        let group = reader.get_row_group(rg)?;
        for row in group.get_row_iter(proj.clone())? {
            f(row?.to_json_value())?;
        }
    }
    Ok(())
}

/// Reads up to `max_rows` rows (projected to `columns`, all when empty) of a local parquet file.
pub fn read_parquet(path: &Path, columns: &[&str], max_rows: usize) -> Result<Vec<Value>> {
    let reader =
        SerializedFileReader::new(File::open(path).with_context(|| format!("open {path:?}"))?)?;
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

/// The `rel="next"` target of an HTTP `Link` header.
fn next_link(h: &str) -> Option<String> {
    h.split(',').find_map(|part| {
        let (target, params) = part.split_once(';')?;
        params.contains("rel=\"next\"").then(|| {
            target
                .trim()
                .trim_start_matches('<')
                .trim_end_matches('>')
                .to_string()
        })
    })
}

/// Hex digest of a file: sha256 of the content, or the git blob sha1 (`blob <len>\0<content>`).
fn file_digest(path: &Path, git_blob: bool) -> Result<String> {
    use ring::digest::{Context as Digest, SHA1_FOR_LEGACY_USE_ONLY, SHA256};
    let len = fs::metadata(path)?.len();
    let mut ctx = if git_blob {
        let mut c = Digest::new(&SHA1_FOR_LEGACY_USE_ONLY);
        c.update(format!("blob {len}\0").as_bytes());
        c
    } else {
        Digest::new(&SHA256)
    };
    let mut f = File::open(path)?;
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        ctx.update(&buf[..n]);
    }
    Ok(ctx
        .finish()
        .as_ref()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// Checks a local file against the size and hash of its listing entry.
pub fn verify_file(path: &Path, file: &RepoFile) -> Result<()> {
    let len = fs::metadata(path)?.len();
    if len != file.size {
        bail!("{}: {len} bytes, expected {}", file.path, file.size);
    }
    let (want, got) = match (&file.sha256, &file.git_sha1) {
        (Some(h), _) => (h, file_digest(path, false)?),
        (None, Some(h)) => (h, file_digest(path, true)?),
        (None, None) => bail!("{}: the Hub publishes no hash", file.path),
    };
    if &got != want {
        bail!("{}: hash {got}, expected {want}", file.path);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_and_dataset_urls() {
        assert_eq!(
            resolve_repo_url(RepoKind::Model, "a/b", "main", "encoder/config.json"),
            "https://huggingface.co/a/b/resolve/main/encoder/config.json"
        );
        assert_eq!(
            resolve_repo_url(RepoKind::Dataset, "a/b", "refs/pr/1", "x.parquet"),
            "https://huggingface.co/datasets/a/b/resolve/refs%2Fpr%2F1/x.parquet"
        );
    }

    #[test]
    fn link_header() {
        assert_eq!(
            next_link(r#"<https://h/a?cursor=x>; rel="next""#).as_deref(),
            Some("https://h/a?cursor=x")
        );
        assert_eq!(
            next_link(r#"<https://h/p>; rel="prev", <https://h/n>; rel="next""#).as_deref(),
            Some("https://h/n")
        );
        assert_eq!(next_link(r#"<https://h/p>; rel="prev""#), None);
    }

    #[test]
    fn digests_match_git_and_sha256() {
        let dir = std::env::temp_dir().join(format!("sky-data-digest-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join("hello.txt");
        fs::write(&p, b"hello\n").unwrap();
        // `printf 'hello\n' | git hash-object --stdin`, and `| shasum -a 256`.
        assert_eq!(
            file_digest(&p, true).unwrap(),
            "ce013625030ba8dba906f756967f9e9ca394464a"
        );
        let mut f = RepoFile {
            path: "hello.txt".into(),
            size: 6,
            sha256: Some("5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03".into()),
            git_sha1: None,
        };
        verify_file(&p, &f).unwrap();
        f.sha256 = Some("00".into());
        assert!(verify_file(&p, &f).is_err());
        f.size = 7;
        assert!(verify_file(&p, &f).is_err());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resolve_url_escapes_refs() {
        assert_eq!(
            resolve_url("a/b", "refs/convert/parquet", "x/y.parquet"),
            "https://huggingface.co/datasets/a/b/resolve/refs%2Fconvert%2Fparquet/x/y.parquet"
        );
    }

    #[test]
    fn sidecar_name() {
        assert_eq!(
            ranges_path(Path::new("/tmp/a.parquet")),
            PathBuf::from("/tmp/a.parquet.ranges")
        );
    }
}
