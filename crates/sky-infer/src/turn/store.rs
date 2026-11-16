//! The on-disk feature store of s6-ear v2 (docs/s6-ear2-prereg.md §3.4, §4.5): written by
//! sky-data (`fetch-turn`, `turn-test`, `turn-eotbench`) and `scripts/turn-teacher.py`, read by
//! sky-train's loader and `sky-turn`. A store is a directory of parts, each with a tag:
//!
//! * `feat-<tag>.bin`: the raw hops of every clip of the part, `[hops][43]` float16 little-endian,
//!   clip after clip (at most the last 400 hops of each clip);
//! * `manifest-<tag>.jsonl`: one line per clip, in the order of the file: `id`, `hops`, `offset`
//!   (bytes into `feat-<tag>.bin`) and the clip's metadata (`split`, `label`, `lang`, `dataset`,
//!   ...). It is written last: a part without its manifest is not finished;
//! * `teacher-<tag>.bin`: the teacher's targets (§5.1), one fixed-size record per manifest line
//!   in the same order ([`TEACHER_RECORD`]); `teacher-<tag>.done` marks it finished.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use serde_json::Value;

use super::{TEACHER_DIM, TEACHER_STEPS};
use crate::audio::TURN_CHANNELS;
use crate::{Error, Result};

/// Bytes of one stored hop.
pub const HOP_BYTES: usize = TURN_CHANNELS * 2;
/// Bytes of one teacher record: the logit (f32), the pooled embedding (384 × f16), the 25 × 384
/// pooled encoder states as u8 codes (`(code − 128) · scale`), then their 25 scales (f32).
pub const TEACHER_RECORD: usize =
    4 + TEACHER_DIM * 2 + TEACHER_STEPS * TEACHER_DIM + TEACHER_STEPS * 4;
/// The zero point of the u8 state codes.
pub const STATE_ZERO: f32 = 128.0;

pub fn feat_name(tag: &str) -> String {
    format!("feat-{tag}.bin")
}
pub fn manifest_name(tag: &str) -> String {
    format!("manifest-{tag}.jsonl")
}
pub fn teacher_name(tag: &str) -> String {
    format!("teacher-{tag}.bin")
}
pub fn teacher_done_name(tag: &str) -> String {
    format!("teacher-{tag}.done")
}

/// Float16 bytes of raw hops.
pub fn encode_hops(hops: &[f32]) -> Vec<u8> {
    hops.iter()
        .flat_map(|&v| sky_gguf::f32_to_f16(v).to_le_bytes())
        .collect()
}

/// Raw hops from their float16 bytes.
pub fn decode_hops(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| sky_gguf::f16_to_f32(u16::from_le_bytes(*b)))
        .collect()
}

/// One clip of a store.
#[derive(Clone, Debug, PartialEq)]
pub struct Clip {
    pub tag: String,
    /// Line of the manifest (the teacher record's index).
    pub index: usize,
    pub id: String,
    pub hops: usize,
    pub offset: u64,
    /// The whole manifest line (`split`, `label`, `lang`, `dataset`, ...).
    pub meta: Value,
}

impl Clip {
    pub fn split(&self) -> &str {
        self.meta["split"].as_str().unwrap_or("")
    }

    /// The label (`true`: the turn is finished), when the manifest carries one.
    pub fn label(&self) -> Option<bool> {
        self.meta["label"].as_bool()
    }
}

/// The teacher's targets of one clip, decoded.
#[derive(Clone, Debug, PartialEq)]
pub struct Teacher {
    pub logit: f32,
    /// `[384]`.
    pub pooled: Vec<f32>,
    /// `[25][384]`, the last step is the end of the window.
    pub states: Vec<f32>,
}

/// Decodes one [`TEACHER_RECORD`].
pub fn decode_teacher(b: &[u8]) -> Teacher {
    let logit = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    let mut at = 4;
    let pooled = decode_hops(&b[at..at + TEACHER_DIM * 2]);
    at += TEACHER_DIM * 2;
    let codes = &b[at..at + TEACHER_STEPS * TEACHER_DIM];
    at += TEACHER_STEPS * TEACHER_DIM;
    let scales: Vec<f32> = b[at..at + TEACHER_STEPS * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|s| f32::from_le_bytes(*s))
        .collect();
    let states = codes
        .iter()
        .enumerate()
        .map(|(i, &c)| (c as f32 - STATE_ZERO) * scales[i / TEACHER_DIM])
        .collect();
    Teacher {
        logit,
        pooled,
        states,
    }
}

/// Encodes one teacher record (the inverse of [`decode_teacher`], for tests and tools).
pub fn encode_teacher(logit: f32, pooled: &[f32], states: &[f32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(TEACHER_RECORD);
    o.extend_from_slice(&logit.to_le_bytes());
    o.extend(encode_hops(&pooled[..TEACHER_DIM]));
    let mut scales = Vec::with_capacity(TEACHER_STEPS);
    for s in states[..TEACHER_STEPS * TEACHER_DIM]
        .as_chunks::<TEACHER_DIM>()
        .0
        .iter()
    {
        let m = s.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let sc = if m > 0.0 { m / 127.0 } else { 1.0 };
        scales.push(sc);
        o.extend(
            s.iter()
                .map(|v| (v / sc + STATE_ZERO).round().clamp(0.0, 255.0) as u8),
        );
    }
    for sc in scales {
        o.extend_from_slice(&sc.to_le_bytes());
    }
    o
}

#[cfg(unix)]
fn read_at(f: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    f.read_exact_at(buf, offset)
}

#[cfg(not(unix))]
fn read_at(f: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = f.try_clone()?;
    f.seek(SeekFrom::Start(offset))?;
    f.read_exact(buf)
}

/// A store opened for reading: every finished part of a directory.
pub struct Store {
    pub dir: PathBuf,
    pub clips: Vec<Clip>,
    feats: HashMap<String, File>,
    teachers: HashMap<String, File>,
}

impl Store {
    /// Every part of `dir` with a manifest, in tag order.
    pub fn open(dir: &Path) -> Result<Store> {
        let mut tags: Vec<String> = std::fs::read_dir(dir)
            .map_err(|e| Error::Format(format!("{}: {e}", dir.display())))?
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let n = e.file_name().to_string_lossy().into_owned();
                n.strip_prefix("manifest-")
                    .and_then(|r| r.strip_suffix(".jsonl"))
                    .map(str::to_string)
            })
            .collect();
        tags.sort();
        let mut clips = Vec::new();
        let (mut feats, mut teachers) = (HashMap::new(), HashMap::new());
        for tag in tags {
            let man = dir.join(manifest_name(&tag));
            let f = File::open(&man)?;
            for (index, line) in BufReader::new(f).lines().enumerate() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let meta: Value = serde_json::from_str(&line)
                    .map_err(|e| Error::Format(format!("{}: {e}", man.display())))?;
                let id = meta["id"].as_str().unwrap_or("").to_string();
                let hops = meta["hops"].as_u64().unwrap_or(0) as usize;
                let offset = meta["offset"].as_u64().unwrap_or(0);
                clips.push(Clip {
                    tag: tag.clone(),
                    index,
                    id,
                    hops,
                    offset,
                    meta,
                });
            }
            feats.insert(tag.clone(), File::open(dir.join(feat_name(&tag)))?);
            if dir.join(teacher_done_name(&tag)).exists() {
                teachers.insert(tag.clone(), File::open(dir.join(teacher_name(&tag)))?);
            }
        }
        Ok(Store {
            dir: dir.to_path_buf(),
            clips,
            feats,
            teachers,
        })
    }

    /// The raw hops `[hops][43]` of a clip.
    pub fn hops(&self, c: &Clip) -> Result<Vec<f32>> {
        let f = self
            .feats
            .get(&c.tag)
            .ok_or_else(|| Error::Missing(feat_name(&c.tag)))?;
        let mut buf = vec![0u8; c.hops * HOP_BYTES];
        read_at(f, &mut buf, c.offset)?;
        Ok(decode_hops(&buf))
    }

    /// True when the teacher's targets of a part are written.
    pub fn has_teacher(&self, tag: &str) -> bool {
        self.teachers.contains_key(tag)
    }

    /// The teacher's targets of a clip (`None` when its part has none).
    pub fn teacher(&self, c: &Clip) -> Result<Option<Teacher>> {
        let Some(f) = self.teachers.get(&c.tag) else {
            return Ok(None);
        };
        let mut buf = vec![0u8; TEACHER_RECORD];
        read_at(f, &mut buf, (c.index * TEACHER_RECORD) as u64)?;
        Ok(Some(decode_teacher(&buf)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn teacher_records_round_trip() {
        let pooled: Vec<f32> = (0..TEACHER_DIM).map(|i| (i as f32 * 0.37).sin()).collect();
        let states: Vec<f32> = (0..TEACHER_STEPS * TEACHER_DIM)
            .map(|i| (i as f32 * 0.011).cos() * (1 + i / TEACHER_DIM) as f32)
            .collect();
        let b = encode_teacher(-1.25, &pooled, &states);
        assert_eq!(b.len(), TEACHER_RECORD);
        let t = decode_teacher(&b);
        assert_eq!(t.logit, -1.25);
        assert!(t
            .pooled
            .iter()
            .zip(&pooled)
            .all(|(a, b)| (a - b).abs() < 1e-3));
        for (s, (a, b)) in t
            .states
            .chunks(TEACHER_DIM)
            .zip(states.chunks(TEACHER_DIM))
            .enumerate()
        {
            let m = b.iter().fold(0.0f32, |x, v| x.max(v.abs()));
            let worst = a
                .iter()
                .zip(b)
                .fold(0.0f32, |x, (u, v)| x.max((u - v).abs()));
            assert!(worst <= m / 127.0 * 0.51, "step {s}: {worst}");
        }
    }

    #[test]
    fn a_store_reads_back_its_parts() {
        let dir = std::env::temp_dir().join(format!("turn-store-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let a: Vec<f32> = (0..3 * TURN_CHANNELS)
            .map(|i| i as f32 * 0.25 - 4.0)
            .collect();
        let b: Vec<f32> = (0..TURN_CHANNELS).map(|i| -(i as f32)).collect();
        let mut feat = encode_hops(&a);
        let off = feat.len();
        feat.extend(encode_hops(&b));
        std::fs::write(dir.join(feat_name("00001")), feat).unwrap();
        std::fs::write(
            dir.join(manifest_name("00001")),
            format!(
                "{{\"id\":\"a\",\"hops\":3,\"offset\":0,\"split\":\"train\",\"label\":true}}\n{{\"id\":\"b\",\"hops\":1,\"offset\":{off},\"split\":\"val\",\"label\":false}}\n"
            ),
        )
        .unwrap();
        // a part without its manifest is not read
        std::fs::write(dir.join(feat_name("00002")), [0u8; 8]).unwrap();
        let s = Store::open(&dir).unwrap();
        assert_eq!(s.clips.len(), 2);
        assert_eq!(
            (s.clips[1].split(), s.clips[1].label()),
            ("val", Some(false))
        );
        assert_eq!(s.hops(&s.clips[0]).unwrap(), a);
        assert_eq!(s.hops(&s.clips[1]).unwrap(), b);
        assert!(s.teacher(&s.clips[0]).unwrap().is_none());
        let rec = encode_teacher(
            0.5,
            &[0.0; TEACHER_DIM],
            &[0.0; TEACHER_STEPS * TEACHER_DIM],
        );
        let mut both = rec.clone();
        both.extend(encode_teacher(
            2.0,
            &[0.0; TEACHER_DIM],
            &[0.0; TEACHER_STEPS * TEACHER_DIM],
        ));
        std::fs::write(dir.join(teacher_name("00001")), both).unwrap();
        std::fs::write(dir.join(teacher_done_name("00001")), "{}").unwrap();
        let s = Store::open(&dir).unwrap();
        assert_eq!(s.teacher(&s.clips[1]).unwrap().unwrap().logit, 2.0);
        std::fs::remove_dir_all(&dir).ok();
    }
}
