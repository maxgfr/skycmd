//! s6-ear v2 data (docs/s6-ear2-prereg.md §3, §4.5, §8.4): feature stores of the student's raw
//! hops (`sky_infer::audio::TurnFrontEnd`, the code the engine runs; format `sky_infer::turn::store`)
//! for the training shards, v1's test WAVs and the eot-bench cuts.
//!
//! * `turn-test`: v1's test WAVs (`data/ear/clips/smart-turn.jsonl`, split `test`) → `test/`: the
//!   hops of each clip and `hashes.jsonl` (id, sha256 of the 16-bit PCM), the inputs of the
//!   leakage filter. No label is written.
//! * `fetch-turn`: `pipecat-ai/smart-turn-data-v3.2-train` one parquet shard at a time (§3.1):
//!   streamed download (hash-checked), the leakage filter of §3.2 (a clip whose id is a test id,
//!   or whose 16-bit PCM hashes to a test clip's, is dropped and counted), the 5% validation split
//!   by a hash of the id (seed 1234), the hops, `manifest-<shard>.jsonl` and `leak-<shard>.json`.
//!   The shard stays on disk until `scripts/turn-teacher.py` has written its targets; `--drop`
//!   removes the shards whose targets are written.
//! * `turn-eotbench`: `livekit/eot-bench-data` (config `all`) one file at a time: every labelled
//!   silence span (`eot` or `hold`) of every turn is one decision on the audio cut at the span's
//!   start (§8.4).
//!
//! The disk rule of §3.4: before each download and every 256 MB of it, the free space of the
//! data folder must stay above `--min-free-gb` (8 by default) plus what the step will write; a
//! step that would cross it stops cleanly with [`DiskLow`] (exit status 3) and resumes where it
//! stopped on the next run (a finished part is never redone).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::{Field, Row};
use serde_json::{json, Value};
use sky_infer::audio::{self as dsp, TurnFrontEnd, SAMPLE_RATE, TURN_WINDOW_HOPS};
use sky_infer::turn::store::{
    encode_hops, feat_name, manifest_name, teacher_done_name, HOP_BYTES, TEACHER_RECORD,
};

use crate::audio::{
    audio_bytes, decode_16k, field_bool, field_str, split_of, tail_pcm, MAX_FRAMES,
};
use crate::hf::{verify_file, Hf, RepoFile, RepoKind};

/// The training release (§3.1), pinned at the commit of the build.
pub const TRAIN_REPO: &str = "pipecat-ai/smart-turn-data-v3.2-train";
pub const TRAIN_REVISION: &str = "f8c95d284595defb6b88a715c356cbc5c31b7ad1";
/// eot-bench (§3.1, §8.4): config `all`, revision `main` (its commit is recorded).
pub const EOT_REPO: &str = "livekit/eot-bench-data";
pub const EOT_REVISION: &str = "main";
/// The validation split: 5% by a hash of the id, seed 1234 (§3.1).
pub const VAL_SEED: u64 = 1234;
pub const VAL_SHARE: f64 = 0.05;

/// A step stopped by the disk rule (exit status 3 in `sky-data`).
#[derive(Debug)]
pub struct DiskLow(pub String);

impl std::fmt::Display for DiskLow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "stopped by the disk rule: {}", self.0)
    }
}

impl std::error::Error for DiskLow {}

/// Free bytes of the file system holding `path` (`df -Pk`).
pub fn free_bytes(path: &Path) -> Result<u64> {
    let out = std::process::Command::new("df")
        .arg("-Pk")
        .arg(path)
        .output()
        .context("running df")?;
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text
        .lines()
        .nth(1)
        .ok_or_else(|| anyhow!("df: no output"))?;
    let avail: u64 = line
        .split_whitespace()
        .nth(3)
        .ok_or_else(|| anyhow!("df: {line}"))?
        .parse()
        .context("df: available")?;
    Ok(avail * 1024)
}

/// The disk rule: the free space of `path` minus what a step will still write stays above
/// `floor` bytes.
#[derive(Clone, Debug)]
pub struct Guard {
    pub path: PathBuf,
    pub floor: u64,
    /// Overrides the measured free space (tests).
    pub fake_free: Option<u64>,
}

impl Guard {
    pub fn new(path: &Path, floor_gb: f64) -> Self {
        Guard {
            path: path.to_path_buf(),
            floor: (floor_gb * (1u64 << 30) as f64) as u64,
            fake_free: None,
        }
    }

    pub fn free(&self) -> Result<u64> {
        match self.fake_free {
            Some(f) => Ok(f),
            None => free_bytes(&self.path),
        }
    }

    /// Fails with [`DiskLow`] unless `need` more bytes keep the free space above the floor.
    pub fn check(&self, need: u64, what: &str) -> Result<()> {
        let free = self.free()?;
        if free < self.floor.saturating_add(need) {
            return Err(DiskLow(format!(
                "{what}: {:.2} GB free, {:.2} GB needed above the floor of {:.2} GB",
                free as f64 / 1e9,
                need as f64 / 1e9,
                self.floor as f64 / 1e9
            ))
            .into());
        }
        Ok(())
    }
}

/// Downloads a repository file to `dest` (kept if it already has the published hash), checking
/// the disk rule before and every 256 MB, then the hash.
pub fn download_guarded(
    hf: &Hf,
    repo: &str,
    rev: &str,
    file: &RepoFile,
    dest: &Path,
    guard: &Guard,
    reserve: u64,
) -> Result<()> {
    if dest.exists() && verify_file(dest, file).is_ok() {
        return Ok(());
    }
    guard.check(file.size + reserve, &format!("download of {}", file.path))?;
    if let Some(d) = dest.parent() {
        fs::create_dir_all(d)?;
    }
    let url = crate::hf::resolve_repo_url(RepoKind::Dataset, repo, rev, &file.path);
    let tmp = dest.with_extension("part");
    let result = (|| -> Result<()> {
        let mut r = hf.stream(&url)?;
        let mut out = File::create(&tmp)?;
        let mut buf = vec![0u8; 8 << 20];
        let (mut done, mut next_check) = (0u64, 256u64 << 20);
        loop {
            let n = r.read(&mut buf)?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])?;
            done += n as u64;
            if done > file.size + (1 << 20) {
                bail!("{}: more than the published {} bytes", file.path, file.size);
            }
            if done >= next_check {
                guard.check(
                    file.size.saturating_sub(done) + reserve,
                    &format!("download of {}", file.path),
                )?;
                next_check += 256 << 20;
            }
        }
        out.flush()?;
        verify_file(&tmp, file)
    })();
    if let Err(e) = result {
        fs::remove_file(&tmp).ok();
        return Err(e);
    }
    fs::rename(&tmp, dest)?;
    Ok(())
}

/// sha256 (hex) of 16 kHz samples as 16-bit PCM, quantized as `sky_infer::audio::write_wav`
/// (v1's test WAVs).
pub fn pcm16_sha256(pcm: &[f32]) -> String {
    let wav = dsp::write_wav(pcm, SAMPLE_RATE);
    sha256_hex(&wav[44..])
}

pub(crate) fn sha256_hex(b: &[u8]) -> String {
    ring::digest::digest(&ring::digest::SHA256, b)
        .as_ref()
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

/// The data chunk of a 16-bit PCM WAV file (as `write_wav` writes them).
pub(crate) fn wav_data(bytes: &[u8]) -> Result<&[u8]> {
    let mut i = 12usize;
    while i + 8 <= bytes.len() {
        let len =
            u32::from_le_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]]) as usize;
        if &bytes[i..i + 4] == b"data" {
            return Ok(&bytes[i + 8..(i + 8 + len).min(bytes.len())]);
        }
        i += 8 + len + (len & 1);
    }
    bail!("no data chunk")
}

/// Writes one part of a store: hops appended to `feat-<tag>.bin`, the manifest written last.
pub(crate) struct PartWriter {
    dir: PathBuf,
    tag: String,
    feat: std::io::BufWriter<File>,
    offset: u64,
    lines: Vec<String>,
}

impl PartWriter {
    pub(crate) fn new(dir: &Path, tag: &str) -> Result<Self> {
        fs::create_dir_all(dir)?;
        Ok(PartWriter {
            dir: dir.to_path_buf(),
            tag: tag.to_string(),
            feat: std::io::BufWriter::new(File::create(dir.join(feat_name(tag)))?),
            offset: 0,
            lines: Vec::new(),
        })
    }

    pub(crate) fn put(&mut self, hops: &[f32], n: usize, mut meta: Value) -> Result<()> {
        self.feat.write_all(&encode_hops(hops))?;
        meta["hops"] = json!(n);
        meta["offset"] = json!(self.offset);
        self.offset += (n * HOP_BYTES) as u64;
        self.lines.push(serde_json::to_string(&meta)?);
        Ok(())
    }

    pub(crate) fn finish(mut self) -> Result<usize> {
        self.feat.flush()?;
        let tmp = self.dir.join(format!("{}.tmp", manifest_name(&self.tag)));
        let mut f = File::create(&tmp)?;
        for l in &self.lines {
            writeln!(f, "{l}")?;
        }
        f.flush()?;
        fs::rename(&tmp, self.dir.join(manifest_name(&self.tag)))?;
        Ok(self.lines.len())
    }
}

/// Runs `f` over `items` on `threads` threads, each with its own front end; results in order.
pub(crate) fn par_map<T: Sync, U: Send>(
    items: &[T],
    threads: usize,
    f: impl Fn(&mut TurnFrontEnd, &T) -> U + Sync,
) -> Vec<U> {
    let threads = threads.max(1).min(items.len().max(1));
    let per = items.len().div_ceil(threads);
    std::thread::scope(|s| {
        let handles: Vec<_> = items
            .chunks(per.max(1))
            .map(|chunk| {
                let f = &f;
                s.spawn(move || {
                    let mut fe = TurnFrontEnd::new();
                    chunk.iter().map(|x| f(&mut fe, x)).collect::<Vec<U>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("worker"))
            .collect()
    })
}

// ----------------------------------------------------------------------------- turn-test

/// Options of `turn-test`.
#[derive(Clone, Debug)]
pub struct TestOptions {
    /// v1's clip manifest (`data/ear/clips/smart-turn.jsonl`); WAV paths are relative to its
    /// parent's parent (`data/ear`).
    pub clips: PathBuf,
    /// The data folder (writes `test/`).
    pub out: PathBuf,
    pub threads: usize,
    /// Test clips read (0: all; a dry run).
    pub max: usize,
    pub guard: Guard,
}

/// `turn-test`: the hops and the PCM hash of every test clip. Reads only `id` and `wav` of each
/// test line of v1's manifest.
pub fn turn_test(o: &TestOptions) -> Result<Value> {
    let dir = o.out.join("test");
    if dir.join(manifest_name("test")).exists() && dir.join("hashes.jsonl").exists() {
        return Ok(json!({"test": "done"}));
    }
    let base = o
        .clips
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| anyhow!("{}: no data folder above it", o.clips.display()))?
        .to_path_buf();
    let mut items: Vec<(String, PathBuf)> = Vec::new();
    for line in BufReader::new(File::open(&o.clips)?).lines() {
        let v: Value = serde_json::from_str(&line?)?;
        if v["split"].as_str() != Some("test") {
            continue;
        }
        if let (Some(id), Some(wav)) = (v["id"].as_str(), v["wav"].as_str()) {
            items.push((id.to_string(), base.join(wav)));
        }
        if o.max > 0 && items.len() >= o.max {
            break;
        }
    }
    // the features take at most 400 hops of 86 bytes a clip
    o.guard.check(
        (items.len() * TURN_WINDOW_HOPS * HOP_BYTES) as u64,
        "turn-test",
    )?;
    let mut part = PartWriter::new(&dir, "test")?;
    let mut hashes = String::new();
    let mut unreadable = 0usize;
    for chunk in items.chunks(2048) {
        let out = par_map(chunk, o.threads, |fe, (id, wav)| -> Result<_> {
            let bytes = fs::read(wav).with_context(|| wav.display().to_string())?;
            let hash = sha256_hex(wav_data(&bytes)?);
            let (pcm, rate) =
                dsp::read_wav(&bytes).map_err(|e| anyhow!("{}: {e}", wav.display()))?;
            let pcm = dsp::resample(&pcm, rate, SAMPLE_RATE);
            let (h, n) = fe.hops(&pcm, TURN_WINDOW_HOPS);
            Ok((id.clone(), hash, h, n))
        });
        for r in out {
            match r {
                Ok((id, hash, h, n)) => {
                    part.put(&h, n, json!({"id": id, "split": "test"}))?;
                    hashes.push_str(&format!("{}\n", json!({"id": id, "sha256": hash})));
                }
                Err(e) => {
                    eprintln!("[turn-test] {e:#}");
                    unreadable += 1;
                }
            }
        }
    }
    fs::write(dir.join("hashes.jsonl"), hashes)?;
    let n = part.finish()?;
    let rep = json!({"test_clips": n, "unreadable": unreadable, "source": o.clips});
    fs::write(
        dir.join("REPORT.json"),
        serde_json::to_string_pretty(&rep)? + "\n",
    )?;
    Ok(rep)
}

/// The test ids and PCM hashes written by `turn-test`.
pub fn test_keys(data: &Path) -> Result<(HashSet<String>, HashSet<String>)> {
    let p = data.join("test/hashes.jsonl");
    let f = File::open(&p).with_context(|| format!("{} (run turn-test first)", p.display()))?;
    let (mut ids, mut hashes) = (HashSet::new(), HashSet::new());
    for line in BufReader::new(f).lines() {
        let v: Value = serde_json::from_str(&line?)?;
        if let (Some(i), Some(h)) = (v["id"].as_str(), v["sha256"].as_str()) {
            ids.insert(i.to_string());
            hashes.insert(h.to_string());
        }
    }
    if ids.is_empty() {
        bail!("{}: no test clip", p.display());
    }
    Ok((ids, hashes))
}

// ----------------------------------------------------------------------------- fetch-turn

/// Options of `fetch-turn`.
#[derive(Clone, Debug)]
pub struct FetchOptions {
    /// The data folder: `shards/` (parquet, transient), `train/` (the store).
    pub out: PathBuf,
    /// Shards to process (all when empty).
    pub shards: Vec<usize>,
    /// Clips read per shard (0: all; a dry run).
    pub max_clips: usize,
    pub threads: usize,
    /// Only remove the shards whose teacher targets are written.
    pub drop_only: bool,
    pub guard: Guard,
}

/// What the filter of §3.2 did to one shard.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ShardReport {
    pub rows: usize,
    pub kept: usize,
    pub train: usize,
    pub val: usize,
    pub dropped_test_id: usize,
    pub dropped_test_pcm: usize,
    pub undecodable: usize,
    /// Kept clips shorter than one 40 ms hop span (no position: never trained on).
    pub no_hops: usize,
}

/// The kept / dropped decision of one decoded clip (§3.2).
#[derive(Debug, PartialEq, Eq)]
pub enum Leak {
    Keep,
    TestId,
    TestPcm,
}

/// §3.2: a training clip is dropped if its id is a test id, or if the hash of its 16-bit PCM (the
/// tail v1 wrote for the test clips) equals a test clip's.
pub fn leak_rule(id: &str, pcm: &[f32], ids: &HashSet<String>, hashes: &HashSet<String>) -> Leak {
    if ids.contains(id) {
        return Leak::TestId;
    }
    if hashes.contains(&pcm16_sha256(tail_pcm(pcm, MAX_FRAMES))) {
        return Leak::TestPcm;
    }
    Leak::Keep
}

pub(crate) fn shard_tag(i: usize) -> String {
    format!("{i:05}")
}

/// The local path of a training shard.
pub fn shard_path(out: &Path, i: usize) -> PathBuf {
    out.join("shards")
        .join(format!("train-{}.parquet", shard_tag(i)))
}

/// Rows of one parquet row group.
pub(crate) fn group_rows(path: &Path, rg: usize) -> Result<Vec<Row>> {
    let reader = SerializedFileReader::new(File::open(path)?)?;
    let g = reader.get_row_group(rg)?;
    g.get_row_iter(None)?
        .map(|r| r.map_err(Into::into))
        .collect()
}

pub(crate) fn row_groups(path: &Path) -> Result<usize> {
    Ok(SerializedFileReader::new(File::open(path)?)?.num_row_groups())
}

/// The features of one downloaded shard (`train/manifest-<tag>.jsonl`, `leak-<tag>.json`).
pub fn shard_features(
    path: &Path,
    out: &Path,
    tag: &str,
    keys: &(HashSet<String>, HashSet<String>),
    max_clips: usize,
    threads: usize,
) -> Result<ShardReport> {
    let dir = out.join("train");
    let mut part = PartWriter::new(&dir, tag)?;
    let mut rep = ShardReport::default();
    let mut row = 0usize;
    'groups: for rg in 0..row_groups(path)? {
        let rows = group_rows(path, rg)?;
        let mut items = Vec::with_capacity(rows.len());
        for r in &rows {
            if max_clips > 0 && rep.rows >= max_clips {
                break;
            }
            let meta = json!({
                "id": field_str(r, "id"),
                "row": row,
                "label": field_bool(r, "endpoint_bool"),
                "lang": field_str(r, "language"),
                "dataset": field_str(r, "dataset"),
                "synthetic": field_bool(r, "synthetic"),
                "midfiller": field_bool(r, "midfiller"),
                "endfiller": field_bool(r, "endfiller"),
            });
            items.push((meta, audio_bytes(r, "audio")));
            row += 1;
            rep.rows += 1;
        }
        let results = par_map(&items, threads, |fe, (meta, bytes)| {
            let id = meta["id"].as_str().unwrap_or("");
            let (Some(b), false, true) = (bytes, id.is_empty(), meta["label"].is_boolean()) else {
                return None;
            };
            let pcm = decode_16k(b).ok()?;
            match leak_rule(id, &pcm, &keys.0, &keys.1) {
                Leak::Keep => {}
                l => return Some(Err(l)),
            }
            Some(Ok(fe.hops(&pcm, TURN_WINDOW_HOPS)))
        });
        for ((meta, _), r) in items.into_iter().zip(results) {
            match r {
                None => rep.undecodable += 1,
                Some(Err(Leak::TestId)) => rep.dropped_test_id += 1,
                Some(Err(_)) => rep.dropped_test_pcm += 1,
                Some(Ok((h, n))) => {
                    let id = meta["id"].as_str().unwrap_or("").to_string();
                    let split = if split_of(&id, VAL_SEED, 0.0, VAL_SHARE) == "val" {
                        "val"
                    } else {
                        "train"
                    };
                    let mut meta = meta;
                    meta["split"] = json!(split);
                    part.put(&h, n, meta)?;
                    rep.kept += 1;
                    if split == "val" {
                        rep.val += 1;
                    } else {
                        rep.train += 1;
                    }
                    rep.no_hops += (n == 0) as usize;
                }
            }
        }
        if max_clips > 0 && rep.rows >= max_clips {
            break 'groups;
        }
    }
    part.finish()?;
    fs::write(
        dir.join(format!("leak-{tag}.json")),
        serde_json::to_string_pretty(&rep)? + "\n",
    )?;
    Ok(rep)
}

/// The leakage counts of every finished shard, summed (§3.2: recorded before training).
pub fn leak_summary(out: &Path) -> Result<Value> {
    let dir = out.join("train");
    let mut total = ShardReport::default();
    let mut shards = BTreeMap::new();
    if dir.exists() {
        for e in fs::read_dir(&dir)? {
            let p = e?.path();
            let name = p
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            let Some(tag) = name
                .strip_prefix("leak-")
                .and_then(|r| r.strip_suffix(".json"))
            else {
                continue;
            };
            let r: ShardReport = serde_json::from_str(&fs::read_to_string(&p)?)?;
            total.rows += r.rows;
            total.kept += r.kept;
            total.train += r.train;
            total.val += r.val;
            total.dropped_test_id += r.dropped_test_id;
            total.dropped_test_pcm += r.dropped_test_pcm;
            total.undecodable += r.undecodable;
            total.no_hops += r.no_hops;
            shards.insert(tag.to_string(), dir.join(teacher_done_name(tag)).exists());
        }
    }
    let v = json!({
        "repo": TRAIN_REPO, "revision": TRAIN_REVISION,
        "shards_done": shards.len(), "teacher_done": shards.values().filter(|&&d| d).count(),
        "total": total,
        "rule": "docs/s6-ear2-prereg.md §3.2: dropped if the id is a test id, or if the sha256 of the 16-bit PCM of the last 128 x 80 ms frames (v1's test WAV tails) equals a test clip's",
        "sentence_overlap": "not checkable: spoken_text is empty in both sets",
    });
    if dir.exists() {
        fs::write(
            dir.join("LEAK.json"),
            serde_json::to_string_pretty(&v)? + "\n",
        )?;
    }
    Ok(v)
}

/// `fetch-turn`.
pub fn fetch_turn(hf: &Hf, o: &FetchOptions) -> Result<Value> {
    let files: Vec<RepoFile> = hf
        .repo_files(RepoKind::Dataset, TRAIN_REPO, TRAIN_REVISION)?
        .into_iter()
        .filter(|f| f.path.starts_with("data/train-") && f.path.ends_with(".parquet"))
        .collect();
    let mut files = files;
    files.sort_by(|a, b| a.path.cmp(&b.path));
    let shards: Vec<usize> = if o.shards.is_empty() {
        (0..files.len()).collect()
    } else {
        o.shards.clone()
    };
    let dir = o.out.join("train");
    fs::create_dir_all(&dir)?;
    let keys = if o.drop_only {
        (HashSet::new(), HashSet::new())
    } else {
        test_keys(&o.out)?
    };
    for &i in &shards {
        let file = files
            .get(i)
            .ok_or_else(|| anyhow!("shard {i}: the release has {} shards", files.len()))?;
        let tag = shard_tag(i);
        let path = shard_path(&o.out, i);
        let features_done = dir.join(manifest_name(&tag)).exists();
        let teacher_done = dir.join(teacher_done_name(&tag)).exists();
        if o.drop_only {
            if teacher_done && path.exists() {
                fs::remove_file(&path)?;
                eprintln!("[fetch-turn] shard {tag}: targets written, parquet removed");
            }
            continue;
        }
        if features_done {
            if teacher_done && path.exists() {
                fs::remove_file(&path)?;
            }
            continue;
        }
        // the shard, the features (at most 400 hops a clip) and the teacher's records
        let clips = (file.size / 120_000).max(1);
        let reserve = clips * (TURN_WINDOW_HOPS * HOP_BYTES + TEACHER_RECORD) as u64;
        download_guarded(
            hf,
            TRAIN_REPO,
            TRAIN_REVISION,
            file,
            &path,
            &o.guard,
            reserve,
        )?;
        o.guard
            .check(reserve, &format!("features of shard {tag}"))?;
        let r = shard_features(&path, &o.out, &tag, &keys, o.max_clips, o.threads)?;
        eprintln!(
            "[fetch-turn] shard {tag}: {} rows, {} kept ({} val), dropped {} by test id and {} by test PCM, {} undecodable",
            r.rows, r.kept, r.val, r.dropped_test_id, r.dropped_test_pcm, r.undecodable
        );
        leak_summary(&o.out)?;
    }
    leak_summary(&o.out)
}

// ----------------------------------------------------------------------------- turn-eotbench

/// Options of `turn-eotbench`.
#[derive(Clone, Debug)]
pub struct EotOptions {
    /// The data folder: `eotbench/parquet/` (transient), `eotbench/store/`.
    pub out: PathBuf,
    /// Files to process, by index in the listing (all when empty).
    pub files: Vec<usize>,
    /// Turns read per file (0: all; a dry run).
    pub max_turns: usize,
    pub threads: usize,
    /// Only remove the parquet files whose Smart Turn pass is written (`smartturn-<tag>.done`;
    /// in span mode, `smartturn200-<tag>.done` as well).
    pub drop_only: bool,
    /// s6-ear v3 (docs/s6-ear3-prereg.md §8.3, §16): one record per span ([`span_hops`]) instead
    /// of one cut at the span's start.
    pub spans: bool,
    /// The revision of eot-bench (v2: `main`, its commit recorded; v3: v2's commit).
    pub revision: String,
    pub guard: Guard,
}

/// v3's span records (docs/s6-ear3-prereg.md §8.3 and amendment 1): the hops of a span from its
/// start − 4.0 s (the window of v2's cut, for the harness check of §8.1) to its start + `dmax`,
/// the largest ask of 0, 0.2, …, 2.4 s that fits inside the span (`start + d ≤ end`, in samples)
/// and inside the audio. `cut` and `end` are samples, `len` the audio's; hop `j` reads
/// `[160 j, 160 j + 640)`. Returns `(first hop, hops stored, cut_hops, dmax_ms)`, `cut_hops` the
/// stored hops whose span ends at or before the cut.
pub fn span_hops(cut: usize, end: usize, len: usize) -> (usize, usize, usize, u64) {
    let upto = |x: usize| dsp::turn_hop_count(x.min(len));
    let c = upto(cut);
    let first = c.saturating_sub(TURN_WINDOW_HOPS);
    let mut d = if end >= cut {
        ((end - cut) / 3200 * 200).min(2400) as u64
    } else {
        0
    };
    while d > 0 && cut + d as usize * 16 > len {
        d -= 200;
    }
    let last = upto(cut + d as usize * 16);
    (first, last - first, c - first, d)
}

/// The cuts of one turn (§8.4): every span labelled `eot` or `hold`, the audio cut at its start.
/// Returns `(span index, label, start, end, cut sample)`.
pub fn cuts(spans: &Value) -> Vec<(usize, bool, f64, f64, usize)> {
    spans
        .as_array()
        .map(|a| {
            a.iter()
                .enumerate()
                .filter_map(|(k, s)| {
                    let eot = match s["label"].as_str()? {
                        "eot" => true,
                        "hold" => false,
                        _ => return None,
                    };
                    let start = s["start"].as_f64()?;
                    let end = s["end"].as_f64().unwrap_or(start);
                    let cut = (start * SAMPLE_RATE as f64).round().max(0.0) as usize;
                    Some((k, eot, start, end, cut))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn field_json(row: &Row, col: &str) -> Value {
    row.get_column_iter()
        .find(|(n, _)| n.as_str() == col)
        .map(|(_, f): (_, &Field)| f.to_json_value())
        .unwrap_or(Value::Null)
}

/// `turn-eotbench`.
pub fn turn_eotbench(hf: &Hf, o: &EotOptions) -> Result<Value> {
    let commit = hf.repo_commit(RepoKind::Dataset, EOT_REPO, &o.revision)?;
    let mut files: Vec<RepoFile> = hf
        .repo_files(RepoKind::Dataset, EOT_REPO, &commit)?
        .into_iter()
        .filter(|f| f.path.starts_with("data/") && f.path.ends_with(".parquet"))
        .collect();
    files.sort_by(|a, b| a.path.cmp(&b.path));
    let base = o.out.join("eotbench");
    let store = base.join("store");
    fs::create_dir_all(&store)?;
    let pick: Vec<usize> = if o.files.is_empty() {
        (0..files.len()).collect()
    } else {
        o.files.clone()
    };
    let mut report = BTreeMap::new();
    for &i in &pick {
        let f = files
            .get(i)
            .ok_or_else(|| anyhow!("file {i}: eot-bench has {} files", files.len()))?;
        // data/<lang>/validation-0000k-of-0000n.parquet -> <lang>-0000k
        let lang = f.path.split('/').nth(1).unwrap_or("x");
        let k = f
            .path
            .rsplit('/')
            .next()
            .and_then(|n| n.strip_prefix("validation-"))
            .and_then(|n| n.split('-').next())
            .unwrap_or("00000");
        let tag = format!("{lang}-{k}");
        let path = base.join("parquet").join(format!("{tag}.parquet"));
        let st_done = store.join(format!("smartturn-{tag}.done")).exists()
            && (!o.spans || store.join(format!("smartturn200-{tag}.done")).exists());
        if o.drop_only {
            if st_done && path.exists() {
                fs::remove_file(&path)?;
            }
            continue;
        }
        if store.join(manifest_name(&tag)).exists() {
            continue;
        }
        let reserve = f.size / 2;
        download_guarded(hf, EOT_REPO, &commit, f, &path, &o.guard, reserve)?;
        let mut part = PartWriter::new(&store, &tag)?;
        let (mut turns, mut n_cuts, mut no_hops, mut undecodable) =
            (0usize, 0usize, 0usize, 0usize);
        let mut row = 0usize;
        'groups: for rg in 0..row_groups(&path)? {
            let rows = group_rows(&path, rg)?;
            let mut items = Vec::new();
            for r in &rows {
                if o.max_turns > 0 && turns >= o.max_turns {
                    break;
                }
                let meta = json!({
                    "turn": field_str(r, "id"),
                    "row": row,
                    "lang": field_str(r, "language"),
                    "cohort": field_str(r, "cohort"),
                    "file": f.path,
                });
                items.push((
                    meta,
                    audio_bytes(r, "audio"),
                    cuts(&field_json(r, "silence_spans")),
                ));
                row += 1;
                turns += 1;
            }
            let spans = o.spans;
            let results = par_map(&items, o.threads, |fe, (_, bytes, cs)| {
                let pcm = decode_16k(bytes.as_ref()?).ok()?;
                Some(
                    cs.iter()
                        .map(|&(_, _, _, end, cut)| {
                            if !spans {
                                let (h, n) = fe.hops(&pcm[..cut.min(pcm.len())], TURN_WINDOW_HOPS);
                                return (h, n, json!({}));
                            }
                            let end_s = (end * SAMPLE_RATE as f64).round().max(0.0) as usize;
                            let (first, n, cut_hops, dmax) = span_hops(cut, end_s, pcm.len());
                            let mut h = vec![0.0f32; n * dsp::TURN_CHANNELS];
                            for (i, row) in h
                                .as_chunks_mut::<{ dsp::TURN_CHANNELS }>()
                                .0
                                .iter_mut()
                                .enumerate()
                            {
                                let s = (first + i) * dsp::HOP;
                                fe.hop(&pcm[s..s + dsp::TURN_SPAN], row);
                            }
                            let len = end_s.saturating_sub(cut);
                            (
                                h,
                                n,
                                json!({"end_sample": end_s, "dmax_ms": dmax, "cut_hops": cut_hops,
                                "short": len < 3200, "long_hold": len > 80_000}),
                            )
                        })
                        .collect::<Vec<_>>(),
                )
            });
            for ((meta, _, cs), r) in items.iter().zip(results) {
                let Some(hops) = r else {
                    undecodable += 1;
                    continue;
                };
                let turn = meta["turn"].as_str().unwrap_or("");
                for (&(k, eot, start, end, cut), (h, n, extra)) in cs.iter().zip(hops) {
                    let mut m = meta.clone();
                    if let Value::Object(x) = extra {
                        for (key, v) in x {
                            m[key.as_str()] = v;
                        }
                    }
                    if o.spans {
                        // the 5 s rule of §8.3 concerns holds only
                        m["long_hold"] = json!(!eot && m["long_hold"] == true);
                    }
                    m["id"] = json!(format!("{turn}#{k}"));
                    m["span"] = json!(k);
                    m["label"] = json!(eot);
                    m["start"] = json!(start);
                    m["end"] = json!(end);
                    m["cut"] = json!(cut);
                    m["split"] = json!("eotbench");
                    part.put(&h, n, m)?;
                    n_cuts += 1;
                    no_hops += (n == 0) as usize;
                }
            }
            if o.max_turns > 0 && turns >= o.max_turns {
                break 'groups;
            }
        }
        part.finish()?;
        let r = json!({"file": f.path, "turns": turns, "cuts": n_cuts, "no_hops": no_hops, "undecodable": undecodable});
        eprintln!("[turn-eotbench] {tag}: {r}");
        report.insert(tag, r);
    }
    let v = json!({"repo": EOT_REPO, "revision": o.revision, "commit": commit, "spans": o.spans, "files": report});
    fs::write(
        base.join(format!("REPORT-{}.json", chrono_free_stamp())),
        serde_json::to_string_pretty(&v)? + "\n",
    )?;
    Ok(v)
}

/// Seconds since the epoch, for report file names.
fn chrono_free_stamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ----------------------------------------------------------------------------- turn-extend

/// s6-ear v3 (docs/s6-ear3-prereg.md §3.4): the hops that follow a clip's end, 240 of them, the
/// trailing silence of the asks up to 2.4 s.
pub const EXT_HOPS: usize = 240;
/// The fill appended to the clip's PCM: 2.44 s, more than the 240 hops read (2.4 s).
pub const FILL_SAMPLES: usize = 39_040;
/// The quietest share of the clip's 10 ms frames makes its background, at least this many.
pub const FILL_SHARE: f64 = 0.10;
pub const FILL_MIN_FRAMES: usize = 20;
/// The raised-cosine cross-fade between the frames of the fill (2.5 ms).
pub const FILL_FADE: usize = 40;
/// Below this RMS the clip's floor is digital silence and the fill is zeros.
pub const DIGITAL_SILENCE: f64 = 1e-5;

/// The fill of a clip (§3.4 and amendment 1): its 10 ms frames of lowest RMS (the lowest 10%,
/// at least 20, all of them for a shorter clip), joined in an order drawn from a seed of the clip
/// id (seeded permutations, one after the other) with 2.5 ms raised-cosine cross-fades, cut to
/// [`FILL_SAMPLES`]; zeros when the pooled RMS of those frames is below 1e-5. Returns the fill
/// and whether it is zeros.
pub fn fill(pcm: &[f32], id: &str) -> (Vec<f32>, bool) {
    let frames: Vec<&[f32]> = pcm.as_chunks::<160>().0.iter().map(|f| &f[..]).collect();
    if frames.is_empty() {
        return (vec![0.0; FILL_SAMPLES], true);
    }
    let rms: Vec<f64> = frames
        .iter()
        .map(|f| (f.iter().map(|&x| x as f64 * x as f64).sum::<f64>() / 160.0).sqrt())
        .collect();
    let mut order: Vec<usize> = (0..frames.len()).collect();
    order.sort_by(|&a, &b| rms[a].total_cmp(&rms[b]).then(a.cmp(&b)));
    let k = ((frames.len() as f64 * FILL_SHARE) as usize)
        .max(FILL_MIN_FRAMES)
        .min(frames.len());
    let quiet = &order[..k];
    let pooled = (quiet.iter().map(|&i| rms[i] * rms[i]).sum::<f64>() / k as f64).sqrt();
    if pooled < DIGITAL_SILENCE {
        return (vec![0.0; FILL_SAMPLES], true);
    }
    let mut rng = crate::rng::Rng::derived(0, &format!("turn3/fill/{id}"));
    let fade: Vec<f32> = (0..FILL_FADE)
        .map(|i| {
            (0.5 - 0.5 * (std::f64::consts::PI * (i as f64 + 0.5) / FILL_FADE as f64).cos()) as f32
        })
        .collect();
    let mut out: Vec<f32> = Vec::with_capacity(FILL_SAMPLES + 160);
    let mut perm: Vec<usize> = Vec::new();
    while out.len() < FILL_SAMPLES {
        if perm.is_empty() {
            perm = quiet.to_vec();
            rng.shuffle(&mut perm);
            perm.reverse();
        }
        let f = frames[perm.pop().expect("non-empty")];
        if out.is_empty() {
            out.extend_from_slice(f);
            continue;
        }
        let at = out.len() - FILL_FADE;
        for i in 0..FILL_FADE {
            out[at + i] = out[at + i] * (1.0 - fade[i]) + f[i] * fade[i];
        }
        out.extend_from_slice(&f[FILL_FADE..]);
    }
    out.truncate(FILL_SAMPLES);
    (out, false)
}

/// The 240 hops after a clip's end on its PCM extended by [`fill`]: the hops `n .. n + 240` of
/// the extended clip, `n` the clip's own hop count. With the clip's own hops before them, they
/// are the hops of the extended clip, exactly (a hop exists once its whole span has arrived).
/// Returns the hops, whether the fill is zeros, and the clip's last own hop (`None` when the clip
/// is shorter than one span).
pub fn extension_hops(
    fe: &mut TurnFrontEnd,
    pcm: &[f32],
    id: &str,
) -> (Vec<f32>, bool, Option<Vec<f32>>) {
    let n = dsp::turn_hop_count(pcm.len());
    let (f, zeros) = fill(pcm, id);
    let mut ext = Vec::with_capacity(pcm.len() + f.len());
    ext.extend_from_slice(pcm);
    ext.extend_from_slice(&f);
    let mut out = vec![0.0f32; EXT_HOPS * dsp::TURN_CHANNELS];
    for (i, row) in out
        .as_chunks_mut::<{ dsp::TURN_CHANNELS }>()
        .0
        .iter_mut()
        .enumerate()
    {
        let s = (n + i) * dsp::HOP;
        fe.hop(&ext[s..s + dsp::TURN_SPAN], row);
    }
    let last = (n > 0).then(|| {
        let mut row = vec![0.0f32; dsp::TURN_CHANNELS];
        let s = (n - 1) * dsp::HOP;
        fe.hop(&pcm[s..s + dsp::TURN_SPAN], &mut row);
        row
    });
    (out, zeros, last)
}

/// Options of `turn-extend`.
#[derive(Clone, Debug)]
pub struct ExtendOptions {
    /// v3's data folder: `shards/` (transient), `st-ext/`, `st-ext-test/`.
    pub out: PathBuf,
    /// v2's data folder: `train/` (its manifests give the kept clips and their rows), `test/`.
    pub v2: PathBuf,
    /// Shards to process (all when empty).
    pub shards: Vec<usize>,
    pub threads: usize,
    /// The test clips instead (v1's WAVs, named by `clips`), into `st-ext-test/`.
    pub test: bool,
    pub clips: PathBuf,
    pub guard: Guard,
}

/// v2's stored last hop of each clip of a part: id → its f16 bytes.
fn v2_last_hops(dir: &Path, tag: &str) -> Result<HashMap<String, Vec<u8>>> {
    let man = dir.join(manifest_name(tag));
    let mut feat = File::open(dir.join(feat_name(tag)))
        .with_context(|| format!("{} (v2's store)", dir.display()))?;
    let mut out = HashMap::new();
    for line in BufReader::new(File::open(&man)?).lines() {
        let v: Value = serde_json::from_str(&line?)?;
        let (n, off) = (
            v["hops"].as_u64().unwrap_or(0),
            v["offset"].as_u64().unwrap_or(0),
        );
        if n == 0 {
            continue;
        }
        let mut b = vec![0u8; HOP_BYTES];
        use std::io::{Seek, SeekFrom};
        feat.seek(SeekFrom::Start(off + (n - 1) * HOP_BYTES as u64))?;
        feat.read_exact(&mut b)?;
        out.insert(v["id"].as_str().unwrap_or("").to_string(), b);
    }
    Ok(out)
}

/// What `turn-extend` did to one part.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ExtendReport {
    pub clips: usize,
    pub zero_fill: usize,
    /// Clips whose recomputed last hop is not v2's stored one (the decoding changed).
    pub last_hop_mismatch: usize,
    /// v2's clips without a hop (shorter than 40 ms): no extension.
    pub no_hops: usize,
    pub undecodable: usize,
}

/// Writes the extension of one part, a chunk of decoded clips at a time (the PCM of a whole shard
/// is never held at once).
struct Extender {
    part: PartWriter,
    last: HashMap<String, Vec<u8>>,
    threads: usize,
    rep: ExtendReport,
}

impl Extender {
    fn new(dir: &Path, tag: &str, last: HashMap<String, Vec<u8>>, threads: usize) -> Result<Self> {
        Ok(Extender {
            part: PartWriter::new(dir, tag)?,
            last,
            threads,
            rep: ExtendReport::default(),
        })
    }

    /// Extends the clips `(meta, pcm)` of a chunk.
    fn add(&mut self, items: &[(Value, Option<Vec<f32>>)]) -> Result<()> {
        let out = par_map(items, self.threads, |fe, (meta, pcm)| {
            let id = meta["id"].as_str().unwrap_or("");
            pcm.as_ref().map(|p| extension_hops(fe, p, id))
        });
        let rep = &mut self.rep;
        for ((meta, _), r) in items.iter().zip(out) {
            let Some((h, zeros, own)) = r else {
                rep.undecodable += 1;
                continue;
            };
            let Some(own) = own else {
                rep.no_hops += 1;
                continue;
            };
            let id = meta["id"].as_str().unwrap_or("");
            let matches = self.last.get(id).is_some_and(|b| *b == encode_hops(&own));
            rep.last_hop_mismatch += !matches as usize;
            rep.zero_fill += zeros as usize;
            rep.clips += 1;
            let mut m = json!({"id": id, "split": meta["split"], "zero_fill": zeros, "last_hop_match": matches});
            if let Some(l) = meta.get("label") {
                m["label"] = l.clone();
            }
            self.part.put(&h, EXT_HOPS, m)?;
        }
        Ok(())
    }

    fn finish(self, dir: &Path, tag: &str) -> Result<ExtendReport> {
        self.part.finish()?;
        fs::write(
            dir.join(format!("ext-{tag}.json")),
            serde_json::to_string_pretty(&self.rep)? + "\n",
        )?;
        Ok(self.rep)
    }
}

/// `turn-extend`: the 83 training shards again (v2's kept clips, from v2's manifests), or with
/// `test` v1's test WAVs; one `st-ext` record of 240 hops per clip id.
pub fn turn_extend(hf: &Hf, o: &ExtendOptions) -> Result<Value> {
    if o.test {
        let dir = o.out.join("st-ext-test");
        if dir.join(manifest_name("test")).exists() {
            return Ok(json!({"test": "done"}));
        }
        let base = o
            .clips
            .parent()
            .and_then(Path::parent)
            .ok_or_else(|| anyhow!("{}: no data folder above it", o.clips.display()))?
            .to_path_buf();
        let mut clips = Vec::new();
        for line in BufReader::new(File::open(&o.clips)?).lines() {
            let v: Value = serde_json::from_str(&line?)?;
            if v["split"].as_str() != Some("test") {
                continue;
            }
            if let (Some(id), Some(wav)) = (v["id"].as_str(), v["wav"].as_str()) {
                clips.push((id.to_string(), base.join(wav)));
            }
        }
        o.guard.check(
            (clips.len() * EXT_HOPS * HOP_BYTES) as u64,
            "turn-extend --test",
        )?;
        let last = v2_last_hops(&o.v2.join("test"), "test")?;
        let mut ex = Extender::new(&dir, "test", last, o.threads)?;
        for chunk in clips.chunks(512) {
            // no label is written: the test's labels are read by the gate only
            let items: Vec<(Value, Option<Vec<f32>>)> = chunk
                .iter()
                .map(|(id, wav)| {
                    let pcm = fs::read(wav)
                        .ok()
                        .and_then(|b| dsp::read_wav(&b).ok())
                        .map(|(p, r)| dsp::resample(&p, r, SAMPLE_RATE));
                    (json!({"id": id, "split": "test"}), pcm)
                })
                .collect();
            ex.add(&items)?;
        }
        let rep = ex.finish(&dir, "test")?;
        return Ok(json!({"test": rep}));
    }
    let files: Vec<RepoFile> = {
        let mut f: Vec<RepoFile> = hf
            .repo_files(RepoKind::Dataset, TRAIN_REPO, TRAIN_REVISION)?
            .into_iter()
            .filter(|f| f.path.starts_with("data/train-") && f.path.ends_with(".parquet"))
            .collect();
        f.sort_by(|a, b| a.path.cmp(&b.path));
        f
    };
    let shards: Vec<usize> = if o.shards.is_empty() {
        (0..files.len()).collect()
    } else {
        o.shards.clone()
    };
    let dir = o.out.join("st-ext");
    fs::create_dir_all(&dir)?;
    let mut total = ExtendReport::default();
    for &i in &shards {
        let file = files
            .get(i)
            .ok_or_else(|| anyhow!("shard {i}: the release has {} shards", files.len()))?;
        let tag = shard_tag(i);
        let path = shard_path(&o.out, i);
        if dir.join(manifest_name(&tag)).exists() {
            fs::remove_file(&path).ok();
            continue;
        }
        let v2dir = o.v2.join("train");
        let mut want: HashMap<usize, Value> = HashMap::new();
        for line in BufReader::new(
            File::open(v2dir.join(manifest_name(&tag)))
                .with_context(|| format!("v2's manifest of shard {tag}"))?,
        )
        .lines()
        {
            let v: Value = serde_json::from_str(&line?)?;
            want.insert(v["row"].as_u64().unwrap_or(u64::MAX) as usize, v);
        }
        let reserve = (want.len() * EXT_HOPS * HOP_BYTES) as u64;
        download_guarded(
            hf,
            TRAIN_REPO,
            TRAIN_REVISION,
            file,
            &path,
            &o.guard,
            reserve,
        )?;
        o.guard
            .check(reserve, &format!("extension of shard {tag}"))?;
        let last = v2_last_hops(&v2dir, &tag)?;
        let mut ex = Extender::new(&dir, &tag, last, o.threads)?;
        let mut row = 0usize;
        for rg in 0..row_groups(&path)? {
            // one row group's clips at a time
            let mut items = Vec::new();
            for r in group_rows(&path, rg)? {
                if let Some(m) = want.get(&row) {
                    let id = field_str(&r, "id").unwrap_or_default();
                    if m["id"].as_str() != Some(id.as_str()) {
                        bail!(
                            "shard {tag} row {row}: id {id}, v2's manifest says {}",
                            m["id"]
                        );
                    }
                    let pcm = audio_bytes(&r, "audio").and_then(|b| decode_16k(&b).ok());
                    items.push((
                        json!({"id": m["id"], "split": m["split"], "label": m["label"]}),
                        pcm,
                    ));
                }
                row += 1;
            }
            ex.add(&items)?;
        }
        let rep = ex.finish(&dir, &tag)?;
        eprintln!("[turn-extend] shard {tag}: {rep:?}");
        total.clips += rep.clips;
        total.zero_fill += rep.zero_fill;
        total.last_hop_mismatch += rep.last_hop_mismatch;
        total.no_hops += rep.no_hops;
        total.undecodable += rep.undecodable;
        fs::remove_file(&path)?;
    }
    Ok(json!({"repo": TRAIN_REPO, "revision": TRAIN_REVISION, "this_run": total}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sky_infer::turn::store::Store;

    #[test]
    fn the_leak_rule_matches_v1s_test_wavs() {
        let pcm: Vec<f32> = (0..30_000).map(|i| (i as f32 * 0.01).sin() * 0.4).collect();
        // v1 wrote the test clip as the WAV of its tail; the train copy decodes to the same PCM
        let wav = dsp::write_wav(tail_pcm(&pcm, MAX_FRAMES), SAMPLE_RATE);
        let h = sha256_hex(wav_data(&wav).unwrap());
        assert_eq!(h, pcm16_sha256(tail_pcm(&pcm, MAX_FRAMES)));
        let ids: HashSet<String> = ["t1".to_string()].into();
        let hashes: HashSet<String> = [h].into();
        assert_eq!(leak_rule("t1", &[0.0; 10], &ids, &hashes), Leak::TestId);
        assert_eq!(leak_rule("other", &pcm, &ids, &hashes), Leak::TestPcm);
        let mut moved = pcm.clone();
        moved[29_999] += 0.01;
        assert_eq!(leak_rule("other", &moved, &ids, &hashes), Leak::Keep);
        // a longer train clip with the same last 10.24 s is caught too (a stricter rule)
        let long: Vec<f32> = (0..200_000)
            .map(|i| (i as f32 * 0.01).sin() * 0.4)
            .collect();
        let wav = dsp::write_wav(tail_pcm(&long, MAX_FRAMES), SAMPLE_RATE);
        let hashes: HashSet<String> = [sha256_hex(wav_data(&wav).unwrap())].into();
        assert_eq!(leak_rule("x", &long, &ids, &hashes), Leak::TestPcm);
    }

    #[test]
    fn the_disk_guard_stops_below_the_floor() {
        let mut g = Guard::new(Path::new("."), 8.0);
        g.fake_free = Some(9 << 30);
        assert!(g.check(512 << 20, "small").is_ok());
        let e = g.check(2 << 30, "big").unwrap_err();
        assert!(e.downcast_ref::<DiskLow>().is_some(), "{e}");
        assert!(free_bytes(Path::new(".")).unwrap() > 0);
    }

    #[test]
    fn cuts_are_the_labelled_spans() {
        let s: Value = serde_json::from_str(
            r#"[{"start":1.8,"end":3.0,"label":"eot"},{"start":0.5,"end":0.7,"label":"hold"},{"start":2.0,"end":2.1,"label":"other"}]"#,
        )
        .unwrap();
        let c = cuts(&s);
        assert_eq!(c.len(), 2);
        assert_eq!(c[0], (0, true, 1.8, 3.0, 28_800));
        assert_eq!(c[1], (1, false, 0.5, 0.7, 8_000));
    }

    #[test]
    fn the_extension_is_the_tail_of_the_hops_of_the_extended_pcm() {
        for (secs, id) in [(1.73, "a"), (3.2, "b"), (0.05, "c")] {
            let pcm = dsp::synth_speech(secs, 3, 0.4);
            let mut fe = TurnFrontEnd::new();
            let (ext, zeros, last) = extension_hops(&mut fe, &pcm, id);
            assert!(!zeros);
            let (f, _) = fill(&pcm, id);
            assert_eq!(f.len(), FILL_SAMPLES);
            let mut whole = pcm.clone();
            whole.extend_from_slice(&f);
            let (all, n_all) = fe.hops(&whole, 0);
            let (own, n) = fe.hops(&pcm, 0);
            assert!(n_all >= n + EXT_HOPS, "{id}");
            // the clip's hops are the extended clip's first ones, the extension the next 240,
            // bit for bit
            let c = dsp::TURN_CHANNELS;
            assert_eq!(&all[..n * c], &own[..], "{id}");
            assert_eq!(&all[n * c..(n + EXT_HOPS) * c], &ext[..], "{id}");
            assert_eq!(
                last.as_deref(),
                (n > 0).then(|| &own[(n - 1) * c..]),
                "{id}"
            );
            // the view at d ends with the last hop whose span ends at or before clip end + d
            let d_hops = 120; // d = 1.2 s
            let end = n + d_hops;
            assert!((end - 1) * 160 + 640 <= pcm.len() + 1200 * 16);
            assert!(end * 160 + 640 > pcm.len() + 1200 * 16);
        }
    }

    #[test]
    fn the_fill_is_the_clips_own_background() {
        let mut pcm = dsp::synth_speech(2.0, 1, 0.5);
        // a quiet stretch at the start: the background
        for x in pcm[..4800].iter_mut() {
            *x *= 0.01;
        }
        let (a, za) = fill(&pcm, "x");
        let (b, _) = fill(&pcm, "x");
        let (c, _) = fill(&pcm, "y");
        assert!(!za && a == b && a != c, "seeded by the clip id");
        let rms = |v: &[f32]| {
            (v.iter().map(|&x| x as f64 * x as f64).sum::<f64>() / v.len() as f64).sqrt()
        };
        assert!(rms(&a) < 0.2 * rms(&pcm), "{} {}", rms(&a), rms(&pcm));
        // digital silence gives zeros
        let mut silent = dsp::synth_speech(1.0, 2, 0.5);
        for x in silent[..8000].iter_mut() {
            *x = 0.0;
        }
        let (z, zz) = fill(&silent, "s");
        assert!(zz && z.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn span_records_hold_v2s_cut_and_the_asks_inside_the_span() {
        let len = 240_000;
        // a span of 1.3 s at 5.0 s: asks 0.2 to 1.2 s fit
        let (first, n, cut_hops, dmax) = span_hops(80_000, 80_000 + 20_800, len);
        assert_eq!(dmax, 1200);
        let c = dsp::turn_hop_count(80_000);
        assert_eq!((first, cut_hops), (c - 400, 400));
        assert_eq!(n, 400 + 120);
        // the d = 0 window is v2's cut: the hops of pcm[..cut], at most 400
        let pcm = dsp::synth_speech(15.0, 4, 0.4);
        let mut fe = TurnFrontEnd::new();
        let (v2, v2n) = fe.hops(&pcm[..80_000], TURN_WINDOW_HOPS);
        let (all, _) = fe.hops(&pcm, 0);
        let c43 = dsp::TURN_CHANNELS;
        assert_eq!(
            &all[first * c43..(first + cut_hops) * c43],
            &v2[..v2n * c43]
        );
        // a short span (0.15 s): d = 0 only; a span near the start: fewer hops before the cut
        assert_eq!(span_hops(80_000, 82_400, len).3, 0);
        let (first, _, cut_hops, _) = span_hops(8_000, 60_000, len);
        assert_eq!((first, cut_hops), (0, dsp::turn_hop_count(8_000)));
        // asks capped at 2.4 s, and at the audio's end
        assert_eq!(span_hops(80_000, 230_000, len).3, 2400);
        assert_eq!(span_hops(230_000, 300_000, len).3, 600);
    }

    #[test]
    fn turn_test_writes_hashes_and_hops_without_labels() {
        let dir = std::env::temp_dir().join(format!("turn-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let ear = dir.join("ear");
        fs::create_dir_all(ear.join("wav")).unwrap();
        fs::create_dir_all(ear.join("clips")).unwrap();
        let mut man = String::new();
        for i in 0..3 {
            let pcm = dsp::synth_speech(0.5 + i as f64, i, 0.5);
            fs::write(
                ear.join(format!("wav/c{i}.wav")),
                dsp::write_wav(&pcm, SAMPLE_RATE),
            )
            .unwrap();
            man.push_str(&format!(
                "{}\n",
                json!({"id": format!("c{i}"), "split": "test", "wav": format!("wav/c{i}.wav"), "label": {"endpoint": true}})
            ));
        }
        man.push_str(&format!(
            "{}\n",
            json!({"id": "tr", "split": "train", "mel": "x"})
        ));
        fs::write(ear.join("clips/smart-turn.jsonl"), man).unwrap();
        let mut guard = Guard::new(&dir, 0.0);
        guard.fake_free = Some(1 << 40);
        let o = TestOptions {
            clips: ear.join("clips/smart-turn.jsonl"),
            out: dir.join("data"),
            threads: 2,
            max: 0,
            guard,
        };
        let r = turn_test(&o).unwrap();
        assert_eq!(r["test_clips"], 3);
        let (ids, hashes) = test_keys(&o.out).unwrap();
        assert_eq!((ids.len(), hashes.len()), (3, 3));
        let s = Store::open(&o.out.join("test")).unwrap();
        assert_eq!(s.clips.len(), 3);
        assert!(s
            .clips
            .iter()
            .all(|c| c.label().is_none() && c.split() == "test"));
        let pcm = dsp::synth_speech(2.5, 2, 0.5);
        let (pcm, _) = dsp::read_wav(&dsp::write_wav(&pcm, SAMPLE_RATE)).unwrap();
        let (h, n) = TurnFrontEnd::new().hops(&pcm, TURN_WINDOW_HOPS);
        assert_eq!(s.clips[2].hops, n);
        let back = s.hops(&s.clips[2]).unwrap();
        assert!(back
            .iter()
            .zip(&h)
            .all(|(a, b)| (a - b).abs() <= 1e-3 * b.abs().max(1.0)));
        // the hash is the one of v1's file, so a train copy of c1 is dropped
        let c1 = dsp::synth_speech(1.5, 1, 0.5);
        let (back, _) = dsp::read_wav(&dsp::write_wav(&c1, SAMPLE_RATE)).unwrap();
        assert_eq!(
            leak_rule("copy", &back, &HashSet::new(), &hashes),
            Leak::TestPcm
        );
        // s6-ear v3: the test clips' extension (turn-extend --test, no download), chunk by chunk;
        // every recomputed last hop is v2's stored one
        let hf = Hf::new(dir.join("hf")).unwrap();
        let mut guard = Guard::new(&dir, 0.0);
        guard.fake_free = Some(1 << 40);
        let e = ExtendOptions {
            out: dir.join("v3"),
            v2: o.out.clone(),
            shards: Vec::new(),
            threads: 2,
            test: true,
            clips: o.clips.clone(),
            guard,
        };
        let r = turn_extend(&hf, &e).unwrap();
        assert_eq!(r["test"]["clips"], 3, "{r}");
        assert_eq!(r["test"]["last_hop_mismatch"], 0, "{r}");
        let x = Store::open(&dir.join("v3/st-ext-test")).unwrap();
        assert!(x
            .clips
            .iter()
            .all(|c| c.hops == EXT_HOPS && c.label().is_none()));
        fs::remove_dir_all(&dir).ok();
    }
}
