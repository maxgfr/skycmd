//! s6-ear data (`docs/s6-ear-prereg.md`, `docs/SPEC-S6.md` §16): audio clips → log-mel features
//! (the front-end of `sky_infer::audio`, the one the engine runs), clip manifests, the audio jev
//! (typed questions per clip) and the Mimi pretraining corpus.
//!
//! `sky-data fetch-audio --set <set>` writes, under `--out` (default `data/ear`):
//!
//! * `audio/<set>/<id>.npy`: log-mel hops `[t][40]` (float16) of the clip, cut to its last
//!   `max_frames` 80 ms frames on the frame grid of the clip start (so training sees exactly the
//!   frames `Engine::decide_audio` keeps);
//! * `wav/<set>/<id>.wav`: the 16 kHz PCM of the **test** clips only (the Smart Turn baseline and
//!   `sky-eval` run on the samples, through the deployed PCM path);
//! * `clips/<set>.jsonl`: one [`Clip`] per clip (split, language, label);
//! * `mimi/{train,val}.jsonl` for `--set mimi`: LibriSpeech log-mel joined with the published
//!   Mimi codes (codebook 0) by utterance id.
//!
//! `sky-data audio-jev` turns the manifests into decision records, one per question
//! (`jev/<task>.jsonl`, state = the `.npy` path relative to the records file).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::{Field, Row};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sky_infer::audio::{self as dsp, LogMel, FRAME_SAMPLES, SAMPLE_RATE};

use crate::hf::Hf;
use crate::record::{write_jsonl, DecisionRecord};
use crate::rng::{hash64, Rng};

/// State frames kept per decision clip: 128 × 80 ms = 10.24 s (`configs/s6-ear-nano.json`).
pub const MAX_FRAMES: usize = 128;

/// Pinned sources (license CC-BY-4.0 for every one, see DATA_LICENSES.md).
pub const SMART_TURN: &str = "pipecat-ai/smart-turn-data-v3-train";
/// The official held-out test set of Smart Turn v3.2 (the clips its benchmark is measured on).
pub const SMART_TURN_TEST: &str = "pipecat-ai/smart-turn-data-v3.2-test";
pub const LIBRISPEECH: &str = "openslr/librispeech_asr";
pub const MIMI_CODES: &str = "shangeth/librispeech-mimi-codes";
pub const MSWC: &str = "MLCommons/ml_spoken_words";
pub const SPEECH_COMMANDS_URL: &str =
    "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz";
pub const SPEECH_COMMANDS_TEST_URL: &str =
    "http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz";

/// The 10 words of the standard 12-class Speech Commands benchmark (plus silence and unknown).
pub const SC_WORDS: [&str; 10] = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
];

/// One clip of a manifest (`clips/<set>.jsonl`).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Clip {
    pub id: String,
    pub set: String,
    pub split: String,
    #[serde(default)]
    pub lang: String,
    /// Log-mel `.npy`, relative to the output directory.
    pub mel: String,
    /// 16 kHz WAV (test clips), relative to the output directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wav: Option<String>,
    /// State frames of the kept audio.
    pub frames: usize,
    /// `endpoint` (bool, smart-turn) or `word` (string, keyword sets).
    pub label: Value,
}

// ----------------------------------------------------------------------------- decoding

/// Decodes WAV or FLAC bytes to mono PCM at 16 kHz.
pub fn decode_16k(bytes: &[u8]) -> Result<Vec<f32>> {
    let (pcm, rate) = if bytes.starts_with(b"RIFF") {
        dsp::read_wav(bytes).map_err(|e| anyhow!(e))?
    } else if bytes.starts_with(b"fLaC") {
        decode_flac(bytes)?
    } else {
        bail!("unknown audio encoding (neither WAV nor FLAC)");
    };
    Ok(dsp::resample(&pcm, rate, SAMPLE_RATE))
}

fn decode_flac(bytes: &[u8]) -> Result<(Vec<f32>, usize)> {
    let mut r = claxon::FlacReader::new(std::io::Cursor::new(bytes))?;
    let info = r.streaminfo();
    let (ch, bits) = (info.channels as usize, info.bits_per_sample);
    let scale = 1.0 / (1u64 << (bits - 1)) as f32;
    let mut mono = Vec::with_capacity(info.samples.unwrap_or(0) as usize);
    let mut acc = 0.0f32;
    for (i, s) in r.samples().enumerate() {
        acc += s? as f32 * scale;
        if (i + 1) % ch == 0 {
            mono.push(acc / ch as f32);
            acc = 0.0;
        }
    }
    Ok((mono, info.sample_rate as usize))
}

/// The samples whose frames are the last `max_frames` frames of `pcm` on the grid of its start
/// (what `Engine::decide_audio` keeps).
pub fn tail_pcm(pcm: &[f32], max_frames: usize) -> &[f32] {
    let n = pcm.len() / FRAME_SAMPLES;
    &pcm[n.saturating_sub(max_frames) * FRAME_SAMPLES..]
}

/// Split of a clip by a seeded hash of its id: `test` with probability `test`, `val` with
/// `val`, else `train` (disjoint by clip).
pub fn split_of(id: &str, seed: u64, test: f64, val: f64) -> &'static str {
    let u = (hash64(format!("ear/{seed}/{id}").as_bytes()) >> 11) as f64 / (1u64 << 53) as f64;
    if u < test {
        "test"
    } else if u < test + val {
        "val"
    } else {
        "train"
    }
}

fn safe_id(id: &str) -> String {
    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Writes the features (and, with `keep_wav`, the samples) of one clip; `None` when it is
/// shorter than one frame.
#[allow(clippy::too_many_arguments)]
pub fn write_clip(
    out: &Path,
    lm: &LogMel,
    set: &str,
    id: &str,
    pcm: &[f32],
    max_frames: usize,
    split: &str,
    lang: &str,
    keep_wav: bool,
    label: Value,
) -> Result<Option<Clip>> {
    let pcm = if max_frames > 0 {
        tail_pcm(pcm, max_frames)
    } else {
        pcm
    };
    let frames = pcm.len() / FRAME_SAMPLES;
    if frames == 0 {
        return Ok(None);
    }
    let name = safe_id(id);
    let mel_rel = format!("audio/{set}/{name}.npy");
    let mel = lm.log_mel(pcm);
    let path = out.join(&mel_rel);
    fs::create_dir_all(path.parent().expect("parent"))?;
    fs::write(
        &path,
        dsp::write_npy_f16(&mel, mel.len() / dsp::N_MELS, dsp::N_MELS),
    )?;
    let wav = if keep_wav {
        let rel = format!("wav/{set}/{name}.wav");
        let p = out.join(&rel);
        fs::create_dir_all(p.parent().expect("parent"))?;
        fs::write(&p, dsp::write_wav(pcm, SAMPLE_RATE))?;
        Some(rel)
    } else {
        None
    };
    Ok(Some(Clip {
        id: id.to_string(),
        set: set.to_string(),
        split: split.to_string(),
        lang: lang.to_string(),
        mel: mel_rel,
        wav,
        frames,
        label,
    }))
}

// ----------------------------------------------------------------------------- parquet audio

/// The `bytes` of an HF `Audio` struct column, or a plain binary column.
fn audio_bytes(row: &Row, col: &str) -> Option<Vec<u8>> {
    for (name, f) in row.get_column_iter() {
        if name != col {
            continue;
        }
        return match f {
            Field::Bytes(b) => Some(b.data().to_vec()),
            Field::Group(g) => g
                .get_column_iter()
                .find_map(|(n, x)| match (n.as_str(), x) {
                    ("bytes", Field::Bytes(b)) => Some(b.data().to_vec()),
                    _ => None,
                }),
            _ => None,
        };
    }
    None
}

fn field_str(row: &Row, col: &str) -> Option<String> {
    row.get_column_iter()
        .find_map(|(n, f)| match (n.as_str() == col, f) {
            (true, Field::Str(s)) => Some(s.clone()),
            _ => None,
        })
}

fn field_bool(row: &Row, col: &str) -> Option<bool> {
    row.get_column_iter()
        .find_map(|(n, f)| match (n.as_str() == col, f) {
            (true, Field::Bool(b)) => Some(*b),
            _ => None,
        })
}

/// Rows of a local parquet file, one row group at a time.
fn for_each_row(path: &Path, mut f: impl FnMut(&Row) -> Result<bool>) -> Result<()> {
    let reader = SerializedFileReader::new(File::open(path).with_context(|| format!("{path:?}"))?)?;
    for rg in 0..reader.num_row_groups() {
        for row in reader.get_row_group(rg)?.get_row_iter(None)? {
            if !f(&row?)? {
                return Ok(());
            }
        }
    }
    Ok(())
}

/// Options of `fetch-audio`.
#[derive(Clone, Debug)]
pub struct FetchAudio {
    pub out: PathBuf,
    /// Clips kept per set (0: all).
    pub max_clips: usize,
    pub seed: u64,
    /// Remove each downloaded shard once processed.
    pub drop_shards: bool,
    /// MSWC: archives read per language and split (each holds ~650 MB of clips).
    pub mswc_archives: usize,
    pub mswc_langs: Vec<String>,
}

/// Counts per split.
#[derive(Debug, Default, Clone, Serialize)]
pub struct SetReport {
    pub set: String,
    pub train: usize,
    pub val: usize,
    pub test: usize,
    pub skipped: usize,
}

impl SetReport {
    fn add(&mut self, c: &Option<Clip>) {
        match c.as_ref().map(|c| c.split.as_str()) {
            Some("train") => self.train += 1,
            Some("val") => self.val += 1,
            Some("test") => self.test += 1,
            _ => self.skipped += 1,
        }
    }

    fn total(&self) -> usize {
        self.train + self.val + self.test
    }
}

fn manifest(o: &FetchAudio, set: &str) -> Result<fs::File> {
    let p = o.out.join("clips").join(format!("{set}.jsonl"));
    fs::create_dir_all(p.parent().expect("parent"))?;
    Ok(File::create(p)?)
}

fn put(f: &mut fs::File, c: &Clip) -> Result<()> {
    writeln!(f, "{}", serde_json::to_string(c)?)?;
    Ok(())
}

/// Smart Turn (fin de tour): test = the official held-out test set of v3.2
/// (`smart-turn-data-v3.2-test`, never trained on by Smart Turn either), train / val = the v3
/// train set (5% val by a seeded hash of the clip id), minus any clip whose id is in the test set.
/// `max_clips` caps each of the two parts.
pub fn fetch_smart_turn(hf: &Hf, o: &FetchAudio) -> Result<SetReport> {
    let lm = LogMel::new();
    let mut rep = SetReport {
        set: "smart-turn".into(),
        ..Default::default()
    };
    let mut man = manifest(o, "smart-turn")?;
    let mut test_ids: HashSet<String> = HashSet::new();
    for (repo, is_test) in [(SMART_TURN_TEST, true), (SMART_TURN, false)] {
        let mut kept = 0usize;
        for (i, url) in hf
            .parquet_urls(repo, "default", "train")?
            .iter()
            .enumerate()
        {
            if o.max_clips > 0 && kept >= o.max_clips {
                break;
            }
            let rel = format!("{}/train-{i:05}.parquet", repo.replace('/', "__"));
            let path = hf.download(url, &rel, 2 << 30)?;
            for_each_row(&path, |row| {
                if o.max_clips > 0 && kept >= o.max_clips {
                    return Ok(false);
                }
                let (Some(id), Some(end), Some(bytes)) = (
                    field_str(row, "id"),
                    field_bool(row, "endpoint_bool"),
                    audio_bytes(row, "audio"),
                ) else {
                    rep.skipped += 1;
                    return Ok(true);
                };
                if !is_test && test_ids.contains(&id) {
                    rep.skipped += 1;
                    return Ok(true);
                }
                let Ok(pcm) = decode_16k(&bytes) else {
                    rep.skipped += 1;
                    return Ok(true);
                };
                let split = if is_test {
                    "test"
                } else if split_of(&id, o.seed, 0.0, 0.05) == "val" {
                    "val"
                } else {
                    "train"
                };
                let lang = field_str(row, "language").unwrap_or_default();
                let c = write_clip(
                    &o.out,
                    &lm,
                    "smart-turn",
                    &id,
                    &pcm,
                    MAX_FRAMES,
                    split,
                    &lang,
                    is_test,
                    json!({"endpoint": end, "dataset": field_str(row, "dataset")}),
                )?;
                if let Some(c) = &c {
                    put(&mut man, c)?;
                    kept += 1;
                }
                if is_test {
                    test_ids.insert(id);
                }
                rep.add(&c);
                Ok(true)
            })?;
            eprintln!("[ear] {repo} shard {i}: {kept} clips so far");
            if o.drop_shards {
                fs::remove_file(&path).ok();
            }
        }
    }
    Ok(rep)
}

// ----------------------------------------------------------------------------- tar archives

/// Every regular file of a (gzip) tar stream through `f(path, bytes)`; `f` returns false to stop.
pub fn for_each_tar_file<R: Read>(
    r: R,
    gz: bool,
    mut f: impl FnMut(&str, Vec<u8>) -> Result<bool>,
) -> Result<()> {
    let inner: Box<dyn Read> = if gz {
        Box::new(flate2::read::GzDecoder::new(r))
    } else {
        Box::new(r)
    };
    let mut ar = tar::Archive::new(inner);
    for e in ar.entries()? {
        let mut e = e?;
        if !e.header().entry_type().is_file() {
            continue;
        }
        let path = e.path()?.to_string_lossy().into_owned();
        let mut buf = Vec::with_capacity(e.size() as usize);
        e.read_to_end(&mut buf)?;
        if !f(&path, buf)? {
            break;
        }
    }
    Ok(())
}

/// The 12-option label of a Speech Commands word (`unknown` for the other 25 words).
pub fn sc_label(word: &str) -> &str {
    if SC_WORDS.contains(&word) {
        word
    } else if word == "_silence_" || word == "_background_noise_" {
        "silence"
    } else {
        "unknown"
    }
}

/// Speech Commands v0.02: train / val from the main archive (`validation_list.txt`; the
/// `testing_list.txt` clips are left out), test = the official 12-class test archive. Unknown
/// words are kept with probability 1/6 in train and val (about one in 12 clips, as in the
/// standard benchmark); silence clips are 1 s crops of `_background_noise_`.
pub fn fetch_speech_commands(hf: &Hf, o: &FetchAudio) -> Result<SetReport> {
    let lm = LogMel::new();
    let mut rep = SetReport {
        set: "speech-commands".into(),
        ..Default::default()
    };
    let mut man = manifest(o, "speech-commands")?;
    let mut rng = Rng::derived(o.seed, "ear/speech-commands");
    // cached archive, read twice: the split lists first, then the clips
    let archive = hf.download(
        SPEECH_COMMANDS_URL,
        "speech_commands/speech_commands_v0.02.tar.gz",
        4 << 30,
    )?;
    let (mut val_list, mut test_list) = (HashSet::new(), HashSet::new());
    for_each_tar_file(File::open(&archive)?, true, |path, bytes| {
        match path.trim_start_matches("./") {
            "validation_list.txt" => {
                val_list = String::from_utf8_lossy(&bytes)
                    .lines()
                    .map(str::to_string)
                    .collect()
            }
            "testing_list.txt" => {
                test_list = String::from_utf8_lossy(&bytes)
                    .lines()
                    .map(str::to_string)
                    .collect()
            }
            _ => {}
        }
        Ok(true)
    })?;
    if val_list.is_empty() || test_list.is_empty() {
        bail!("{SPEECH_COMMANDS_URL}: validation_list.txt / testing_list.txt not found");
    }
    let cap = if o.max_clips > 0 {
        o.max_clips
    } else {
        usize::MAX
    };
    for_each_tar_file(File::open(&archive)?, true, |path, bytes| {
        let p = path.trim_start_matches("./");
        if !p.ends_with(".wav") || test_list.contains(p) {
            return Ok(true);
        }
        let word = p.split('/').next().unwrap_or("");
        if word != "_background_noise_" && rep.total() >= cap {
            return Ok(true);
        }
        let mut emit = |id: String, pcm: &[f32], label: &str, split: &str| -> Result<()> {
            let c = write_clip(
                &o.out,
                &lm,
                "speech-commands",
                &id,
                pcm,
                MAX_FRAMES,
                split,
                "en",
                false,
                json!({"word": label, "raw": word}),
            )?;
            if let Some(c) = &c {
                put(&mut man, c)?;
            }
            rep.add(&c);
            Ok(())
        };
        if word == "_background_noise_" {
            // silence: 1 s crops every 0.5 s, 10% of them to val
            let pcm = decode_16k(&bytes)?;
            let starts = (0..pcm.len().saturating_sub(SAMPLE_RATE)).step_by(SAMPLE_RATE / 2);
            for (k, a) in starts.enumerate() {
                let id = format!("{p}#{k}");
                let split = if split_of(&id, o.seed, 0.0, 0.1) == "val" {
                    "val"
                } else {
                    "train"
                };
                emit(id, &pcm[a..a + SAMPLE_RATE], "silence", split)?;
            }
            return Ok(true);
        }
        let label = sc_label(word);
        if label == "unknown" && rng.below(6) != 0 {
            return Ok(true);
        }
        let split = if val_list.contains(p) { "val" } else { "train" };
        emit(p.to_string(), &decode_16k(&bytes)?, label, split)?;
        Ok(true)
    })?;
    // the official test set: folders named after the 12 labels (_silence_, _unknown_ included)
    for_each_tar_file(hf.stream(SPEECH_COMMANDS_TEST_URL)?, true, |path, bytes| {
        let p = path.trim_start_matches("./");
        if !p.ends_with(".wav") {
            return Ok(true);
        }
        let folder = p.split('/').next().unwrap_or("");
        let label = match folder {
            "_silence_" => "silence",
            "_unknown_" => "unknown",
            w => sc_label(w),
        };
        let pcm = decode_16k(&bytes)?;
        let c = write_clip(
            &o.out,
            &lm,
            "speech-commands",
            &format!("test/{p}"),
            &pcm,
            MAX_FRAMES,
            "test",
            "en",
            true,
            json!({"word": label, "raw": folder}),
        )?;
        if let Some(c) = &c {
            put(&mut man, c)?;
        }
        rep.add(&c);
        Ok(true)
    })?;
    Ok(rep)
}

/// MSWC (`data/wav/<lang>/<split>/audio/<n>.tar.gz`, labels in `data/splits/<lang>/splits.tar.gz`):
/// the first `mswc_archives` archives of each split and language. The words are kept as they are
/// (the held-out zero-shot words are chosen by `audio-jev`).
pub fn fetch_mswc(hf: &Hf, o: &FetchAudio) -> Result<SetReport> {
    let lm = LogMel::new();
    let mut rep = SetReport {
        set: "mswc".into(),
        ..Default::default()
    };
    let mut man = manifest(o, "mswc")?;
    let url = |f: &str| crate::hf::resolve_url(MSWC, "main", f);
    for lang in &o.mswc_langs {
        // the split CSVs: LINK,WORD,VALID,SPEAKER,GENDER
        let mut word_of: HashMap<String, String> = HashMap::new();
        for_each_tar_file(
            hf.stream(&url(&format!("data/splits/{lang}/splits.tar.gz")))?,
            true,
            |path, bytes| {
                if path.ends_with(".csv") {
                    for line in String::from_utf8_lossy(&bytes).lines().skip(1) {
                        let mut it = line.split(',');
                        if let (Some(link), Some(word)) = (it.next(), it.next()) {
                            let key = link
                                .trim_end_matches(".opus")
                                .trim_end_matches(".wav")
                                .replace('/', "_");
                            word_of.insert(key, word.to_string());
                        }
                    }
                }
                Ok(true)
            },
        )?;
        for (split, out_split) in [("train", "train"), ("dev", "val"), ("test", "test")] {
            for n in 0..o.mswc_archives {
                let f = format!("data/wav/{lang}/{split}/audio/{n}.tar.gz");
                let mut kept = 0usize;
                for_each_tar_file(hf.stream(&url(&f))?, true, |path, bytes| {
                    if o.max_clips > 0 && kept >= o.max_clips {
                        return Ok(false);
                    }
                    let name = path.rsplit('/').next().unwrap_or(path);
                    let key = name.trim_end_matches(".wav").trim_end_matches(".opus");
                    let Some(word) = word_of
                        .get(key)
                        .cloned()
                        .or_else(|| key.split('_').next().map(str::to_string))
                    else {
                        return Ok(true);
                    };
                    let pcm = match decode_16k(&bytes) {
                        Ok(p) => p,
                        Err(_) => return Ok(true),
                    };
                    let c = write_clip(
                        &o.out,
                        &lm,
                        "mswc",
                        &format!("{lang}/{key}"),
                        &pcm,
                        MAX_FRAMES,
                        out_split,
                        lang,
                        out_split == "test",
                        json!({"word": word}),
                    )?;
                    if let Some(c) = &c {
                        put(&mut man, c)?;
                        kept += 1;
                    }
                    rep.add(&c);
                    Ok(true)
                })?;
                eprintln!("[ear] mswc {lang} {split} archive {n}: {kept} clips");
            }
        }
    }
    Ok(rep)
}

/// LibriSpeech audio joined with the published Mimi codes by utterance id (codebook 0 only):
/// `train.clean.100` → `mimi/train.jsonl`, `validation.clean` (dev_clean) → `mimi/val.jsonl`.
/// The log-mel files keep the whole utterance (pretraining crops windows of it).
pub fn fetch_mimi(hf: &Hf, o: &FetchAudio) -> Result<SetReport> {
    let lm = LogMel::new();
    let mut rep = SetReport {
        set: "mimi".into(),
        ..Default::default()
    };
    let dir = o.out.join("mimi");
    fs::create_dir_all(&dir)?;
    for (ls_split, codes_split, split) in [
        ("validation", "dev_clean", "val"),
        ("train.100", "train_clean_100", "train"),
    ] {
        // codes first (small): id -> codebook 0
        let mut codes: HashMap<String, Vec<u16>> = HashMap::new();
        for (i, url) in hf
            .parquet_urls(MIMI_CODES, "default", codes_split)?
            .iter()
            .enumerate()
        {
            let rel = format!(
                "{}/{codes_split}-{i:05}.parquet",
                MIMI_CODES.replace('/', "__")
            );
            let path = hf.download(url, &rel, 8 << 30)?;
            crate::hf::for_each_parquet_row(&path, &["id", "codes"], |v| {
                if let (Some(id), Some(cb0)) = (
                    v["id"].as_str(),
                    v["codes"].get(0).and_then(Value::as_array),
                ) {
                    codes.insert(
                        id.to_string(),
                        cb0.iter().map(|c| c.as_u64().unwrap_or(0) as u16).collect(),
                    );
                }
                Ok(())
            })?;
        }
        let mut lines = File::create(dir.join(format!("{split}.jsonl")))?;
        let mut n = 0usize;
        for (i, url) in hf
            .parquet_urls(LIBRISPEECH, "clean", ls_split)?
            .iter()
            .enumerate()
        {
            if o.max_clips > 0 && n >= o.max_clips {
                break;
            }
            let rel = format!(
                "{}/clean-{ls_split}-{i:05}.parquet",
                LIBRISPEECH.replace('/', "__")
            );
            let path = hf.download(url, &rel, 4 << 30)?;
            for_each_row(&path, |row| {
                if o.max_clips > 0 && n >= o.max_clips {
                    return Ok(false);
                }
                let (Some(id), Some(bytes)) = (field_str(row, "id"), audio_bytes(row, "audio"))
                else {
                    rep.skipped += 1;
                    return Ok(true);
                };
                let Some(cb0) = codes.get(&id) else {
                    rep.skipped += 1;
                    return Ok(true);
                };
                let pcm = decode_16k(&bytes)?;
                let Some(c) = write_clip(
                    &o.out,
                    &lm,
                    "librispeech",
                    &id,
                    &pcm,
                    0,
                    split,
                    "en",
                    false,
                    Value::Null,
                )?
                else {
                    rep.skipped += 1;
                    return Ok(true);
                };
                writeln!(
                    lines,
                    "{}",
                    json!({"id": id, "mel": format!("../{}", c.mel), "codes": cb0, "split": split, "text": field_str(row, "text")})
                )?;
                n += 1;
                if split == "train" {
                    rep.train += 1;
                } else {
                    rep.val += 1;
                }
                Ok(true)
            })?;
            if o.drop_shards {
                fs::remove_file(&path).ok();
            }
        }
    }
    Ok(rep)
}

// ----------------------------------------------------------------------------- the audio jev

/// Pre-registered question texts (`docs/s6-ear-prereg.md` §2).
pub const Q_TURN: &str = "Has the speaker finished their turn?";
pub const TURN_OPTS: [&str; 2] = [
    "false: the speaker is still talking or will go on",
    "true: the speaker has finished their turn",
];
pub const Q_KEYWORD: &str = "Which word was said?";
pub const SILENCE_OPT: &str = "silence: nothing was said";
pub const UNKNOWN_OPT: &str = "unknown: another word";
pub const Q_DUPLEX: &str = "What should I do now?";
pub const DUPLEX_OPTS: [&str; 4] = [
    "listen: keep listening",
    "backchannel: acknowledge with a short mm-hmm",
    "speak: take the turn and answer",
    "yield: stop talking, I am being interrupted",
];

/// Options of a zero-shot (MSWC) keyword question: the gold word and the distractors.
pub const ZS_OPTIONS: usize = 10;
/// Held-out MSWC words per language (never trained on).
pub const ZS_WORDS: usize = 50;
/// Test questions per held-out word (its first clips, so the gold word is uniform).
pub const ZS_CLIPS_PER_WORD: usize = 20;

fn one_hot(k: usize, i: usize) -> Vec<f32> {
    (0..k).map(|j| (j == i) as u8 as f32).collect()
}

/// A decision record about a clip; `state` is its `.npy` relative to `jev/`.
fn ear_record(
    src: &str,
    c: &Clip,
    t: &str,
    ins: &str,
    opts: Vec<String>,
    target: Vec<f32>,
) -> Value {
    let mut v = serde_json::to_value(DecisionRecord {
        src: src.into(),
        t: t.into(),
        ins: ins.into(),
        opts,
        state: format!("../{}", c.mel),
        target,
        split: c.split.clone(),
    })
    .expect("record");
    if let Some(w) = &c.wav {
        v["wav"] = json!(format!("../{w}"));
    }
    v["lang"] = json!(c.lang);
    v["clip"] = json!(c.id);
    v
}

fn read_clips(path: &Path) -> Result<Vec<Clip>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let f = File::open(path)?;
    let mut out = Vec::new();
    for line in BufReader::new(f).lines() {
        let line = line?;
        if !line.trim().is_empty() {
            out.push(serde_json::from_str(&line).with_context(|| format!("{}", path.display()))?);
        }
    }
    Ok(out)
}

/// Report of `audio-jev`: records per task and split, the held-out words.
#[derive(Debug, Default, Serialize)]
pub struct JevReport {
    pub counts: BTreeMap<String, BTreeMap<String, usize>>,
    pub held_out: BTreeMap<String, Vec<String>>,
}

/// Writes `jev/turn_end.jsonl`, `jev/keyword.jsonl` and `jev/keyword_zs.jsonl` from the clip
/// manifests under `dir`; `about` (GLM labels of transcripts, `label-transcripts`) adds
/// `jev/about.jsonl` when given.
pub fn write_jev(dir: &Path, seed: u64, about: Option<&Path>) -> Result<JevReport> {
    let jev = dir.join("jev");
    fs::create_dir_all(&jev)?;
    let mut rep = JevReport::default();
    let count = |rep: &mut JevReport, task: &str, recs: &[Value]| {
        let e = rep.counts.entry(task.to_string()).or_default();
        for r in recs {
            *e.entry(r["split"].as_str().unwrap_or("?").to_string())
                .or_default() += 1;
        }
    };
    // fin de tour
    let turn: Vec<Value> = read_clips(&dir.join("clips/smart-turn.jsonl"))?
        .iter()
        .filter_map(|c| {
            let end = c.label["endpoint"].as_bool()?;
            Some(ear_record(
                "ear:turn_end",
                c,
                "noul",
                Q_TURN,
                TURN_OPTS.iter().map(|s| s.to_string()).collect(),
                one_hot(2, end as usize),
            ))
        })
        .collect();
    count(&mut rep, "turn_end", &turn);
    write_jsonl(&jev.join("turn_end.jsonl"), &turn)?;
    // mots-clés : the 12-option benchmark
    let mut sc_opts: Vec<String> = SC_WORDS.iter().map(|s| s.to_string()).collect();
    sc_opts.push(SILENCE_OPT.into());
    sc_opts.push(UNKNOWN_OPT.into());
    let kw: Vec<Value> = read_clips(&dir.join("clips/speech-commands.jsonl"))?
        .iter()
        .filter_map(|c| {
            let w = c.label["word"].as_str()?;
            let i = match w {
                "silence" => 10,
                "unknown" => 11,
                w => SC_WORDS.iter().position(|x| *x == w)?,
            };
            Some(ear_record(
                "ear:keyword",
                c,
                "choice",
                Q_KEYWORD,
                sc_opts.clone(),
                one_hot(12, i),
            ))
        })
        .collect();
    count(&mut rep, "keyword", &kw);
    write_jsonl(&jev.join("keyword.jsonl"), &kw)?;
    // mots-clés zéro-shot (MSWC): per language, ZS_WORDS test words with >= ZS_CLIPS_PER_WORD test
    // clips are held out (seeded) and give exactly that many test questions each; train / val keep
    // the other words; each question offers the gold word and ZS_OPTIONS - 1 distractors of the
    // same pool (the held-out pool for the test questions)
    let mswc = read_clips(&dir.join("clips/mswc.jsonl"))?;
    let mut zs = Vec::new();
    let mut by_lang: BTreeMap<String, Vec<&Clip>> = BTreeMap::new();
    for c in &mswc {
        by_lang.entry(c.lang.clone()).or_default().push(c);
    }
    for (lang, clips) in by_lang {
        let mut test_counts: BTreeMap<String, usize> = BTreeMap::new();
        for c in clips.iter().filter(|c| c.split == "test") {
            if let Some(w) = c.label["word"].as_str() {
                *test_counts.entry(w.to_string()).or_default() += 1;
            }
        }
        let mut pool: Vec<String> = test_counts
            .into_iter()
            .filter(|(_, n)| *n >= ZS_CLIPS_PER_WORD)
            .map(|(w, _)| w)
            .collect();
        Rng::derived(seed, &format!("ear/zs/{lang}")).shuffle(&mut pool);
        pool.truncate(ZS_WORDS);
        let held: HashSet<&str> = pool.iter().map(String::as_str).collect();
        let seen: Vec<String> = {
            let mut s: Vec<String> = clips
                .iter()
                .filter_map(|c| c.label["word"].as_str())
                .filter(|w| !held.contains(w))
                .map(str::to_string)
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();
            s.sort();
            s
        };
        rep.held_out.insert(lang.clone(), pool.clone());
        let mut rng = Rng::derived(seed, &format!("ear/zs-opts/{lang}"));
        let mut per_word: HashMap<String, usize> = HashMap::new();
        for c in clips {
            let Some(w) = c.label["word"].as_str() else {
                continue;
            };
            if c.split == "test" && held.contains(w) {
                let k = per_word.entry(w.to_string()).or_default();
                if *k >= ZS_CLIPS_PER_WORD {
                    continue;
                }
                *k += 1;
            }
            let (words, keep) = match (c.split.as_str(), held.contains(w)) {
                ("test", true) => (&pool, true),
                ("test", false) => (&seen, false),
                (_, false) => (&seen, true),
                (_, true) => (&pool, false),
            };
            if !keep || words.len() < ZS_OPTIONS {
                continue;
            }
            let mut opts = vec![w.to_string()];
            while opts.len() < ZS_OPTIONS {
                let d = &words[rng.below(words.len())];
                if !opts.contains(d) {
                    opts.push(d.clone());
                }
            }
            rng.shuffle(&mut opts);
            let gold = opts.iter().position(|o| o == w).expect("gold");
            let k = opts.len();
            zs.push(ear_record(
                &format!("ear:keyword_zs:{lang}"),
                c,
                "choice",
                Q_KEYWORD,
                opts,
                one_hot(k, gold),
            ));
        }
    }
    count(&mut rep, "keyword_zs", &zs);
    write_jsonl(&jev.join("keyword_zs.jsonl"), &zs)?;
    if let Some(labels) = about {
        let about = about_records(dir, labels)?;
        count(&mut rep, "about", &about);
        write_jsonl(&jev.join("about.jsonl"), &about)?;
    }
    fs::write(
        jev.join("REPORT.json"),
        serde_json::to_string_pretty(&rep)? + "\n",
    )?;
    Ok(rep)
}

// ----------------------------------------------------------------------------- about (GLM)

/// The pre-registered `about` questions asked of every transcript (v2): (type, question,
/// options).
pub fn about_questions() -> Vec<(&'static str, &'static str, Vec<&'static str>)> {
    vec![
        (
            "noul",
            "Is the speaker asking a question?",
            vec![
                "false: no question is asked",
                "true: the speaker asks a question",
            ],
        ),
        (
            "noul",
            "Does the speaker express a negative feeling?",
            vec![
                "false: no negative feeling",
                "true: a negative feeling is expressed",
            ],
        ),
        (
            "noul",
            "Is the speaker giving an instruction or a request?",
            vec![
                "false: no instruction or request",
                "true: an instruction or a request",
            ],
        ),
        (
            "choice",
            "What is the speech mostly about?",
            vec![
                "people and relationships",
                "nature, animals or places",
                "work, money or society",
                "religion, history or ideas",
                "everyday actions and objects",
                "none of these",
            ],
        ),
    ]
}

/// One transcript question per (utterance, about question), the state being the transcript
/// (text, as the teacher sees it) and the target uniform (no gold: the teacher decides), with
/// the utterance id kept in `src` so the labels join back to the audio.
pub fn transcript_records(dir: &Path) -> Result<Vec<DecisionRecord>> {
    let mut out = Vec::new();
    for split in ["train", "val"] {
        let p = dir.join("mimi").join(format!("{split}.jsonl"));
        if !p.exists() {
            continue;
        }
        for line in BufReader::new(File::open(&p)?).lines() {
            let v: Value = serde_json::from_str(&line?)?;
            let (Some(id), Some(text)) = (v["id"].as_str(), v["text"].as_str()) else {
                continue;
            };
            for (t, q, opts) in about_questions() {
                out.push(DecisionRecord {
                    src: format!("ear-about:{id}"),
                    t: t.into(),
                    ins: q.into(),
                    opts: opts.iter().map(|s| s.to_string()).collect(),
                    state: text.to_lowercase(),
                    target: vec![1.0 / opts.len() as f32; opts.len()],
                    // the teacher labels train records only: every utterance goes as train, the
                    // audio split is restored in `about_records`
                    split: "train".into(),
                });
            }
        }
    }
    Ok(out)
}

/// The audio records of the `about` task: each GLM label of a transcript question becomes a
/// soft target on the utterance's log-mel (cross-modal distillation: the teacher reads the text,
/// the student hears the audio).
pub fn about_records(dir: &Path, labels: &Path) -> Result<Vec<Value>> {
    let labels: HashMap<String, crate::distill::Label> =
        crate::record::read_jsonl::<crate::distill::Label>(labels)?
            .into_iter()
            .map(|l| (l.id.clone(), l))
            .collect();
    let mut meta: HashMap<String, (String, String)> = HashMap::new();
    for split in ["train", "val"] {
        let p = dir.join("mimi").join(format!("{split}.jsonl"));
        if !p.exists() {
            continue;
        }
        for line in BufReader::new(File::open(&p)?).lines() {
            let v: Value = serde_json::from_str(&line?)?;
            if let (Some(id), Some(mel)) = (v["id"].as_str(), v["mel"].as_str()) {
                meta.insert(
                    id.to_string(),
                    (mel.trim_start_matches("../").to_string(), split.to_string()),
                );
            }
        }
    }
    let mut out = Vec::new();
    for r in transcript_records(dir)? {
        let Some(l) = labels.get(&crate::distill::record_id(&r)) else {
            continue;
        };
        let id = r.src.trim_start_matches("ear-about:");
        let Some((mel, split)) = meta.get(id) else {
            continue;
        };
        let c = Clip {
            id: id.to_string(),
            set: "librispeech".into(),
            split: split.clone(),
            lang: "en".into(),
            mel: mel.clone(),
            wav: None,
            frames: 0,
            label: Value::Null,
        };
        out.push(ear_record(
            "ear:about",
            &c,
            &r.t,
            &r.ins,
            r.opts.clone(),
            l.probs.clone(),
        ));
    }
    Ok(out)
}

// ----------------------------------------------------------------------------- duplex (v2)

/// One speech segment of a meeting: speaker, start and end (seconds), and whether its dialogue
/// act is a backchannel (AMI `backchannel` acts: "mm-hmm", "yeah", …).
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct Segment {
    pub speaker: String,
    pub start: f64,
    pub end: f64,
    #[serde(default)]
    pub backchannel: bool,
}

/// Look-ahead of the duplex rule (pre-registered: 500 ms).
pub const DUPLEX_HORIZON: f64 = 0.5;

/// The `duplex_action` label for agent `me` at time `t` (`docs/s6-ear-prereg.md` §5, v2): index
/// into [`DUPLEX_OPTS`].
///
/// 1. `me` is speaking at `t` and another speaker starts in `(t, t + 500 ms]` while `me` still
///    speaks → **yield** (3) — unless that start is a backchannel;
/// 2. `me` starts a segment in `(t, t + 500 ms]`: a backchannel → **backchannel** (1), else
///    **speak** (2);
/// 3. anything else → **listen** (0).
pub fn duplex_label(segs: &[Segment], me: &str, t: f64) -> usize {
    let within = |x: f64| x > t && x <= t + DUPLEX_HORIZON;
    let speaking = segs
        .iter()
        .any(|s| s.speaker == me && s.start <= t && s.end > t);
    if speaking
        && segs.iter().any(|s| {
            s.speaker != me
                && !s.backchannel
                && within(s.start)
                && segs
                    .iter()
                    .any(|m| m.speaker == me && m.start <= s.start && m.end > s.start)
        })
    {
        return 3;
    }
    let next = segs
        .iter()
        .filter(|s| s.speaker == me && within(s.start))
        .min_by(|a, b| a.start.total_cmp(&b.start));
    match next {
        Some(s) if s.backchannel => 1,
        Some(_) => 2,
        None => 0,
    }
}

/// The rule baseline of the duplex gate: speak after 500 ms of silence of every other speaker,
/// else listen (the classic voice-activity endpointing).
pub fn silence_rule(segs: &[Segment], me: &str, t: f64) -> usize {
    let others_quiet = !segs
        .iter()
        .any(|s| s.speaker != me && s.end > t - DUPLEX_HORIZON && s.start <= t);
    let me_speaking = segs
        .iter()
        .any(|s| s.speaker == me && s.start <= t && s.end > t);
    if others_quiet && !me_speaking {
        2
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(sp: &str, a: f64, b: f64, bc: bool) -> Segment {
        Segment {
            speaker: sp.into(),
            start: a,
            end: b,
            backchannel: bc,
        }
    }

    #[test]
    fn duplex_rule_follows_the_preregistration() {
        // A talks, B (the agent) answers 0.3 s after A stops
        let s = vec![seg("A", 0.0, 3.0, false), seg("B", 3.3, 5.0, false)];
        assert_eq!(duplex_label(&s, "B", 3.0), 2);
        assert_eq!(duplex_label(&s, "B", 2.0), 0); // B starts 1.3 s later: listen
                                                   // B says "mm-hmm" in the middle of A's turn
        let s = vec![seg("A", 0.0, 6.0, false), seg("B", 2.2, 2.6, true)];
        assert_eq!(duplex_label(&s, "B", 2.0), 1);
        // B talks, A cuts in while B still talks: B yields
        let s = vec![seg("B", 0.0, 4.0, false), seg("A", 3.8, 6.0, false)];
        assert_eq!(duplex_label(&s, "B", 3.5), 3);
        // ... but A's backchannel is not an interruption
        let s = vec![seg("B", 0.0, 4.0, false), seg("A", 3.8, 4.1, true)];
        assert_eq!(duplex_label(&s, "B", 3.5), 0);
        // the silence rule speaks after 500 ms without anyone else
        let s = vec![seg("A", 0.0, 3.0, false)];
        assert_eq!(silence_rule(&s, "B", 3.2), 0);
        assert_eq!(silence_rule(&s, "B", 3.6), 2);
    }

    #[test]
    fn splits_tails_and_speech_commands_labels() {
        let n = 10_000;
        let test = (0..n)
            .filter(|i| split_of(&format!("c{i}"), 7, 0.1, 0.05) == "test")
            .count();
        assert!((800..1200).contains(&test), "{test}");
        assert_eq!(split_of("x", 7, 0.1, 0.05), split_of("x", 7, 0.1, 0.05));
        let pcm = vec![0.0f32; 200 * FRAME_SAMPLES + 77];
        let t = tail_pcm(&pcm, 128);
        assert_eq!((t.len() - 77) / FRAME_SAMPLES, 128);
        assert_eq!(
            tail_pcm(&pcm[..5 * FRAME_SAMPLES], 128).len(),
            5 * FRAME_SAMPLES
        );
        assert_eq!(
            (sc_label("yes"), sc_label("bed"), sc_label("_silence_")),
            ("yes", "unknown", "silence")
        );
    }

    #[test]
    fn clips_and_jev_round_trip() {
        let dir = std::env::temp_dir().join(format!("ear-jev-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let lm = LogMel::new();
        let pcm: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        let mut man = Vec::new();
        for (i, end) in [true, false, true].iter().enumerate() {
            let split = if i == 2 { "test" } else { "train" };
            let c = write_clip(
                &dir,
                &lm,
                "smart-turn",
                &format!("st/{i}"),
                &pcm,
                MAX_FRAMES,
                split,
                "fra",
                split == "test",
                json!({"endpoint": end}),
            )
            .unwrap()
            .unwrap();
            assert_eq!(c.frames, 12);
            man.push(c);
        }
        fs::create_dir_all(dir.join("clips")).unwrap();
        write_jsonl(&dir.join("clips/smart-turn.jsonl"), &man).unwrap();
        let rep = write_jev(&dir, 1, None).unwrap();
        assert_eq!(rep.counts["turn_end"]["train"], 2);
        let line = fs::read_to_string(dir.join("jev/turn_end.jsonl")).unwrap();
        let first: Value = serde_json::from_str(line.lines().next().unwrap()).unwrap();
        assert_eq!(first["target"], json!([0.0, 1.0]));
        let mel = dir.join("jev").join(first["state"].as_str().unwrap());
        let (v, rows, cols) = dsp::read_npy(&fs::read(mel).unwrap()).unwrap();
        assert_eq!((rows, cols, v.len()), (100, 40, 4000));
        let test: Value = serde_json::from_str(line.lines().nth(2).unwrap()).unwrap();
        let wav = dir.join("jev").join(test["wav"].as_str().unwrap());
        let back = decode_16k(&fs::read(wav).unwrap()).unwrap();
        assert_eq!(back.len(), 16_000);
        fs::remove_dir_all(&dir).ok();
    }
}
