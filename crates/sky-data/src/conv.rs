//! s6-ear v3 conversation records (docs/s6-ear3-prereg.md §3.2, §3.3, §3.7, §16): real
//! two-speaker conversations of `otoearth/otoSpeech-full-duplex-280h`, one speaker per channel,
//! cut where the deployed policy asks and labelled from the other channel.
//!
//! * `fetch-conv`: the WebDataset shards one at a time (gated: `HF_TOKEN`), every session through
//!   §3.2 in full: each channel decoded on its own and resampled to 16 kHz, its gain, the energy
//!   VAD and the policy replayed ([`crate::turn_policy`]), the other channel's speech runs, the
//!   label rule, every drop counted, the bleed filter, the redactions, DualTurn's `splits.json`,
//!   the caps; the records of A's channel only (`sky_infer::turn::store` format, `conv/`), the
//!   manifest and the counts. The shard is deleted once its records are written. Per shard it
//!   also writes, transiently, the 8 s PCM tails of the validation and test records (Smart Turn's
//!   recorded opinion, `scripts/turn-teacher.py --records`, which removes them) and the 16 kHz
//!   channels of each validation session (the label cross-check, `scripts/turn-vadcheck.py`,
//!   which removes them). `--finalize` makes the store the caps' exact draw and writes `CONV.json`.
//! * `conv-crosscheck`: the label rule again on Silero's speech runs of the validation sessions,
//!   the one-to-one matching of the offsets within 150 ms, `CROSSCHECK.json` (§3.7).
//!
//! **The caps (§3.2 item 8).** "Drawn uniformly without replacement (seed 3)" is made streaming:
//! each eligible record has a key, a seeded hash of its id, and a split keeps the records of its
//! `cap` smallest keys, which is a uniform draw without replacement of `cap` of them. A shard
//! writes only the records whose key is then among the smallest; the parts are compacted as
//! records fall out (docs/s6-ear3-prereg.md amendment 1).
//!
//! Times are 16 kHz sample indices from the session's start; a record's offset `o` is the end of a
//! 10 ms frame, so a multiple of 160.

use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sky_infer::audio::{self as dsp, TurnFrontEnd, HOP, SAMPLE_RATE, TURN_CHANNELS, TURN_SPAN};
use sky_infer::turn::store::{feat_name, manifest_name, HOP_BYTES};

use crate::audio::{decode_flac_channels, for_each_tar_file};
use crate::hf::{Hf, RepoFile, RepoKind};
use crate::rng::hash64;
use crate::turn::{download_guarded, par_map, shard_tag, Guard, PartWriter};
use crate::turn_policy::{self, Pause, FRAME_SAMPLES};

/// The corpus, at the revision of the owner's access (frozen 19/11/2026).
pub const OTO_REPO: &str = "otoearth/otoSpeech-full-duplex-280h";
pub const OTO_REVISION: &str = "4b3c57b8947d5f651dd6b5e825e7f34b205759e5";
pub const OTO_SHARDS: usize = 151;
/// DualTurn's session split (§3.3), the file `splits.json` only.
pub const SPLITS_REPO: &str = "anyreach-ai/dualturn-otospeech-turn-taking";
pub const SPLITS_REVISION: &str = "5b700b4b7fe6a21aa3a5acce5f235913fb40cb2e";
pub const SPLITS_SHA256: &str = "7550074d7b1761d946dcf3f0bb61fbf1372fe97becd68d752ce631a21ef8b436";

/// 16 kHz samples per millisecond.
const MS: u64 = 16;
/// DualTurn's cleaning (§3.2 item 3): speech runs under 150 ms removed, silences under 150 ms
/// filled.
pub const CLEAN_MS: u64 = 150;
/// A backchannel: a run of at most 1.0 s with at least 1.0 s of silence before and after it.
pub const BACKCHANNEL_MS: u64 = 1000;
/// A floor take: a run of at least 1.0 s that is not a backchannel.
pub const TAKE_MS: u64 = 1000;
/// The label horizon (§3.2 item 4).
pub const HORIZON_MS: u64 = 4000;
/// An overlap: a floor take of B under way at o, or begun in [o − 1.0 s, o).
pub const OVERLAP_MS: u64 = 1000;
/// Records whose [o − 4 s, o + 4 s] comes within 0.5 s of a redaction are dropped.
pub const REDACTION_SPAN_MS: u64 = 4000;
pub const REDACTION_PAD_MS: u64 = 500;
/// The bleed filter: a session goes when either channel's ratio is above 2.0.
pub const BLEED_MAX: f64 = 2.0;
/// The gain: the 95th percentile of a channel's 10 ms frame RMS is brought to 0.1.
pub const GAIN_TARGET: f64 = 0.1;
pub const GAIN_PERCENTILE: f64 = 0.95;
/// The caps of §3.2 item 8 and their seed.
pub const CAPS: [(&str, usize); 3] = [("train", 120_000), ("val", 20_000), ("test", 30_000)];
pub const CAP_SEED: u64 = 3;
/// A record keeps A's hops from o + 0.2 − 4.0 s: the 380 hops whose span ends at or before o.
pub const BACK_HOPS: usize = 380;
/// The teacher's input: the last 8 s up to o + 0.2 s (§5).
pub const TAIL_SAMPLES: usize = 8 * SAMPLE_RATE;
/// The first ask, d = 0.2 s.
pub const FIRST_ASK_MS: u64 = 200;
/// Conversation test records kept as 16-bit WAV windows for the WASM parity of §7.
pub const PARITY_RECORDS: usize = 20;
/// Free space a shard may take beyond its own download: its records, tails and WAVs.
pub const SHARD_RESERVE: u64 = 1_500_000_000;
/// The label cross-check (§3.7): a Silero offset ends a run followed by at least 200 ms of
/// silence; offsets match within 150 ms; the two bars.
pub const SILERO_GAP_MS: u64 = 200;
pub const MATCH_MS: u64 = 150;
pub const AGREEMENT_BAR: f64 = 0.85;
pub const UNCONFIRMED_BAR: f64 = 0.30;

// ----------------------------------------------------------------------------- the label rule

/// A run of speech, `[start, end)` in samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Run {
    pub start: u64,
    pub end: u64,
}

impl Run {
    pub fn len(&self) -> u64 {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }
}

/// The runs of consecutive speech frames of `frame` samples each.
pub fn runs_of(speech: &[bool], frame: u64) -> Vec<Run> {
    let mut out = Vec::new();
    let mut start = None;
    for (k, &s) in speech.iter().enumerate() {
        match (s, start) {
            (true, None) => start = Some(k as u64),
            (false, Some(a)) => {
                out.push(Run {
                    start: a * frame,
                    end: k as u64 * frame,
                });
                start = None;
            }
            _ => {}
        }
    }
    if let Some(a) = start {
        out.push(Run {
            start: a * frame,
            end: speech.len() as u64 * frame,
        });
    }
    out
}

/// DualTurn's cleaning: runs shorter than `min` removed, then silences shorter than `min`
/// filled.
pub fn clean(runs: &[Run], min: u64) -> Vec<Run> {
    let mut out: Vec<Run> = Vec::new();
    for r in runs.iter().filter(|r| r.len() >= min) {
        match out.last_mut() {
            Some(l) if r.start - l.end < min => l.end = r.end,
            _ => out.push(*r),
        }
    }
    out
}

/// What a cleaned run of a channel is (§3.2 item 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Kind {
    Backchannel,
    Take,
    Other,
}

/// One channel's cleaned speech runs and their kinds, over `n` samples.
#[derive(Clone, Debug)]
pub struct Channel {
    pub runs: Vec<Run>,
    pub kinds: Vec<Kind>,
    pub n: u64,
}

impl Channel {
    /// Cleans raw runs and classifies them. The silence before the first run and after the last
    /// is the one observed, from the session's start and up to its end.
    pub fn new(raw: &[Run], n: u64) -> Channel {
        let runs = clean(raw, CLEAN_MS * MS);
        let kinds = (0..runs.len())
            .map(|i| {
                let r = runs[i];
                let before = if i == 0 {
                    r.start
                } else {
                    r.start - runs[i - 1].end
                };
                let after = runs
                    .get(i + 1)
                    .map_or(n.saturating_sub(r.end), |x| x.start - r.end);
                if r.len() <= BACKCHANNEL_MS * MS
                    && before >= BACKCHANNEL_MS * MS
                    && after >= BACKCHANNEL_MS * MS
                {
                    Kind::Backchannel
                } else if r.len() >= TAKE_MS * MS {
                    Kind::Take
                } else {
                    Kind::Other
                }
            })
            .collect();
        Channel { runs, kinds, n }
    }

    /// The cleaned run of the utterance that ends at `o`: the run holding the sample just before
    /// `o`, else the last run that ends at or before `o`.
    pub fn utterance_at(&self, o: u64) -> Option<usize> {
        if o == 0 {
            return None;
        }
        let i = self.runs.partition_point(|r| r.start < o);
        if i == 0 {
            return None;
        }
        // runs[i - 1] starts at or before o - 1: it holds it, or it is the last one before o
        Some(i - 1)
    }
}

/// Why a candidate record is dropped (§3.2 item 5, the record rules, in their order).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Drop {
    /// The utterance of A that ends at o is itself a backchannel.
    Backchannel,
    /// B is inside a floor take at o, or one began in [o − 1.0 s, o).
    Overlap,
    /// Neither r nor t_B comes within 4.0 s of o.
    NoFollow,
    /// r and t_B fall on the same sample within 4.0 s: neither "t_B < r" nor "r < t_B" holds.
    Tie,
    /// [o − 4.0 s, o + 4.0 s] comes within 0.5 s of a redacted range.
    Redaction,
}

/// The label of a pause of A at `o` and what decided it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Labelled {
    /// End of turn (`true`) or hold.
    pub eot: bool,
    /// The onset of B's first floor take after o.
    pub tb: Option<u64>,
    /// A's next onset.
    pub resume: Option<u64>,
}

/// §3.2 items 4 and 5 (the record rules): the label of a pause of A at `o`, given A's next onset
/// `resume`, or the first rule that drops it. Both paths of §3.7 call it, the energy path with the
/// policy's onset and the Silero path with the next Silero run.
pub fn label(
    a: &Channel,
    b: &Channel,
    o: u64,
    resume: Option<u64>,
    redactions: &[(u64, u64)],
) -> std::result::Result<Labelled, Drop> {
    if let Some(i) = a.utterance_at(o) {
        if a.kinds[i] == Kind::Backchannel {
            return Err(Drop::Backchannel);
        }
    }
    let mut tb = None;
    for (r, k) in b.runs.iter().zip(&b.kinds) {
        if *k != Kind::Take {
            continue;
        }
        let under_way = r.start < o && r.end > o;
        let just_began = r.start < o && r.start + OVERLAP_MS * MS >= o;
        if under_way || just_began {
            return Err(Drop::Overlap);
        }
        if r.start >= o {
            tb = Some(r.start);
            break;
        }
    }
    let h = o + HORIZON_MS * MS;
    let (t, r) = (tb.unwrap_or(u64::MAX), resume.unwrap_or(u64::MAX));
    let eot = if t <= h && t < r {
        true
    } else if r <= h && r < t {
        false
    } else if t <= h && t == r {
        return Err(Drop::Tie);
    } else {
        return Err(Drop::NoFollow);
    };
    let (lo, hi) = (
        o.saturating_sub(REDACTION_SPAN_MS * MS),
        o + REDACTION_SPAN_MS * MS,
    );
    let pad = REDACTION_PAD_MS * MS;
    if redactions
        .iter()
        .any(|&(s, e)| hi + pad >= s && lo <= e + pad)
    {
        return Err(Drop::Redaction);
    }
    Ok(Labelled { eot, tb, resume })
}

// ----------------------------------------------------------------------------- gain and bleed

/// RMS of each whole 10 ms frame.
pub fn frame_rms(pcm: &[f32]) -> Vec<f64> {
    pcm.as_chunks::<FRAME_SAMPLES>()
        .0
        .iter()
        .map(|f| {
            let s: f64 = f.iter().map(|&x| x as f64 * x as f64).sum();
            (s / FRAME_SAMPLES as f64).sqrt()
        })
        .collect()
}

/// The nearest-rank percentile `q` of `v`.
pub fn percentile(v: &[f64], q: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let k = ((q * s.len() as f64).ceil() as usize).clamp(1, s.len());
    Some(s[k - 1])
}

/// The gain of a channel (§3.2 item 1): `0.1 / p95` of its frame RMS; `None` when that
/// percentile is 0 (a channel silent on more than 95% of its frames cannot be scaled).
pub fn gain(pcm: &[f32]) -> Option<f32> {
    let p = percentile(&frame_rms(pcm), GAIN_PERCENTILE)?;
    (p > 0.0).then(|| (GAIN_TARGET / p) as f32)
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    Some(if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    })
}

/// The bleed ratio of a channel (§3.2 item 5): the median RMS of its frames that are VAD-silent
/// while the other channel speaks, over the median RMS of the frames where both are silent.
/// `None` when either set is empty; 1 when both medians are 0; infinite when only the second is.
pub fn bleed_ratio(rms: &[f64], me: &[bool], other: &[bool]) -> Option<f64> {
    let n = rms.len().min(me.len()).min(other.len());
    let (mut num, mut den) = (Vec::new(), Vec::new());
    for k in 0..n {
        if me[k] {
            continue;
        }
        if other[k] {
            num.push(rms[k]);
        } else {
            den.push(rms[k]);
        }
    }
    let (a, b) = (median(num)?, median(den)?);
    Some(if b > 0.0 {
        a / b
    } else if a > 0.0 {
        f64::INFINITY
    } else {
        1.0
    })
}

// ----------------------------------------------------------------------------- one session

/// What the metadata of a session gives (only these fields are read: no profile, no survey).
#[derive(Clone, Debug, Default)]
pub struct SessionMeta {
    pub id: String,
    pub duration: f64,
    /// Redacted ranges in seconds.
    pub redactions: Vec<(f64, f64)>,
    /// The pseudonymous `speaker_id` of channels 0 and 1.
    pub speakers: [String; 2],
}

/// Reads the session's `json` member.
pub fn parse_meta(bytes: &[u8]) -> Result<SessionMeta> {
    let v: Value = serde_json::from_slice(bytes).context("session metadata")?;
    let id = v["session_id"]
        .as_str()
        .ok_or_else(|| anyhow!("metadata without session_id"))?
        .to_string();
    let mut redactions = Vec::new();
    match &v["redacted_segments"] {
        Value::Null => {}
        Value::Array(a) => {
            for r in a {
                let (s, e) = match (r["start_sec"].as_f64(), r["end_sec"].as_f64()) {
                    (Some(s), Some(e)) => (s, e),
                    _ => bail!("{id}: a redacted segment without start_sec / end_sec: {r}"),
                };
                if !(s.is_finite() && e.is_finite() && s >= 0.0 && e >= s) {
                    bail!("{id}: a redacted segment out of order: {r}");
                }
                redactions.push((s, e));
            }
        }
        x => bail!("{id}: redacted_segments is not a list: {x}"),
    }
    let mut speakers = [String::new(), String::new()];
    for s in v["speakers"].as_array().into_iter().flatten() {
        if let (Some(c), Some(i)) = (s["channel"].as_u64(), s["speaker_id"].as_str()) {
            if c < 2 {
                speakers[c as usize] = i.to_string();
            }
        }
    }
    Ok(SessionMeta {
        id,
        duration: v["duration"].as_f64().unwrap_or(0.0),
        redactions,
        speakers,
    })
}

/// Why a whole session is dropped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionDrop {
    /// A channel's 95th percentile of frame RMS is 0: it cannot be gain-normalised.
    SilentChannel,
    /// The bleed filter (either ratio above 2.0, or undefined).
    Bleed,
}

/// A candidate record: a pause of the policy on channel `channel`, labelled or dropped.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub channel: usize,
    pub pause: Pause,
    pub outcome: std::result::Result<Labelled, Drop>,
}

/// One session through §3.2 items 1–6.
pub struct Processed {
    /// The gain-normalised 16 kHz channels.
    pub pcm: [Vec<f32>; 2],
    pub gains: [Option<f32>; 2],
    pub bleed: [Option<f64>; 2],
    pub drop: Option<SessionDrop>,
    pub candidates: Vec<Candidate>,
    /// The energy VAD's raw speech runs of each channel.
    pub raw_runs: [Vec<Run>; 2],
    /// Samples of the shorter channel.
    pub n: u64,
}

/// §3.2 items 1–6 on the two 16 kHz channels of a session (`redactions` in samples).
pub fn process_session(channels: [Vec<f32>; 2], redactions: &[(u64, u64)]) -> Processed {
    let n = channels[0].len().min(channels[1].len());
    let gains = [gain(&channels[0][..n]), gain(&channels[1][..n])];
    let pcm: [Vec<f32>; 2] = {
        let [a, b] = channels;
        let scale = |mut x: Vec<f32>, g: Option<f32>| {
            x.truncate(n);
            let g = g.unwrap_or(1.0);
            x.iter_mut().for_each(|v| *v *= g);
            x
        };
        [scale(a, gains[0]), scale(b, gains[1])]
    };
    let replays = [turn_policy::replay(&pcm[0]), turn_policy::replay(&pcm[1])];
    let raw_runs = [
        runs_of(&replays[0].speech, FRAME_SAMPLES as u64),
        runs_of(&replays[1].speech, FRAME_SAMPLES as u64),
    ];
    let n64 = n as u64;
    let chans = [
        Channel::new(&raw_runs[0], n64),
        Channel::new(&raw_runs[1], n64),
    ];
    let bleed = [
        bleed_ratio(&replays[0].rms, &replays[0].speech, &replays[1].speech),
        bleed_ratio(&replays[1].rms, &replays[1].speech, &replays[0].speech),
    ];
    let drop = if gains.iter().any(Option::is_none) {
        Some(SessionDrop::SilentChannel)
    } else if bleed.iter().any(|b| b.is_none_or(|r| r > BLEED_MAX)) {
        Some(SessionDrop::Bleed)
    } else {
        None
    };
    let mut candidates = Vec::new();
    for a in 0..2 {
        for p in turn_policy::pauses(&replays[a].events) {
            let outcome = label(&chans[a], &chans[1 - a], p.offset, p.resume, redactions);
            candidates.push(Candidate {
                channel: a,
                pause: p,
                outcome,
            });
        }
    }
    Processed {
        pcm,
        gains,
        bleed,
        drop,
        candidates,
        raw_runs,
        n: n64,
    }
}

// ----------------------------------------------------------------------------- records

/// The manifest line of a record (with `hops` and `offset`, added by the part writer).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Record {
    pub id: String,
    pub session: String,
    pub channel: usize,
    /// The offset in samples and in seconds.
    pub o_sample: u64,
    pub o: f64,
    /// `true`: end of turn.
    pub label: bool,
    /// The largest valid ask (ms).
    pub dmax_ms: u64,
    /// t_B − o and r − o in seconds.
    pub tb: Option<f64>,
    pub r: Option<f64>,
    pub split: String,
    /// The pseudonymous speaker id of A (§3.3, the speaker-unseen subset only).
    pub speaker: String,
    /// Stored hops whose span ends at or before o: the view at d ends `d / 10` hops later.
    pub cut_hops: usize,
    /// The caps' key (a seeded hash of the id).
    pub key: u64,
}

/// The caps' key of a record id.
pub fn record_key(id: &str) -> u64 {
    hash64(format!("conv/{CAP_SEED}/{id}").as_bytes())
}

/// The hop range of a record: `(first, last)` global hop indices, inclusive, and `cut_hops`.
/// Hop `j` reads samples `[160 j, 160 j + 640)`; the last hop of the view at d ends at or before
/// `o + d`.
pub fn record_hops(o: u64, dmax_ms: u64) -> (u64, u64, usize) {
    let last = (o + dmax_ms * MS - TURN_SPAN as u64) / HOP as u64;
    let cut = (o - TURN_SPAN as u64) / HOP as u64; // the last hop ending at or before o
    let first = (cut + 1).saturating_sub(BACK_HOPS as u64);
    (first, last, (cut + 1 - first) as usize)
}

/// The raw hops `[first, last]` of a channel.
fn hops_range(fe: &mut TurnFrontEnd, pcm: &[f32], first: u64, last: u64) -> Vec<f32> {
    let mut out = vec![0.0f32; (last - first + 1) as usize * TURN_CHANNELS];
    for (i, row) in out
        .as_chunks_mut::<TURN_CHANNELS>()
        .0
        .iter_mut()
        .enumerate()
    {
        let s = (first as usize + i) * HOP;
        fe.hop(&pcm[s..s + TURN_SPAN], row);
    }
    out
}

/// `len` samples of `pcm` ending at `end`, zeros on the left where they fall before the start.
fn left_padded(pcm: &[f32], end: usize, len: usize) -> Vec<f32> {
    let end = end.min(pcm.len());
    let start = end.saturating_sub(len);
    let mut out = vec![0.0f32; len - (end - start)];
    out.extend_from_slice(&pcm[start..end]);
    out
}

/// A mono 32-bit float WAV at 16 kHz.
pub fn write_wav_f32(pcm: &[f32]) -> Vec<u8> {
    let data = (pcm.len() * 4) as u32;
    let mut b = Vec::with_capacity(44 + pcm.len() * 4);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&(SAMPLE_RATE as u32).to_le_bytes());
    b.extend_from_slice(&(SAMPLE_RATE as u32 * 4).to_le_bytes());
    b.extend_from_slice(&4u16.to_le_bytes());
    b.extend_from_slice(&32u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data.to_le_bytes());
    for &x in pcm {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

/// The draw of the caps: for each split, the keys of its `cap` smallest eligible records so far.
pub struct Caps {
    heaps: HashMap<String, BinaryHeap<u64>>,
    caps: HashMap<String, usize>,
}

impl Caps {
    pub fn new(caps: &[(&str, usize)]) -> Caps {
        Caps {
            heaps: HashMap::new(),
            caps: caps.iter().map(|&(s, c)| (s.to_string(), c)).collect(),
        }
    }

    /// Adds an eligible record's key; true when it is among the `cap` smallest now.
    pub fn insert(&mut self, split: &str, key: u64) -> bool {
        let cap = self.caps.get(split).copied().unwrap_or(0);
        let h = self.heaps.entry(split.to_string()).or_default();
        if h.len() < cap {
            h.push(key);
            true
        } else if h.peek().is_some_and(|&m| key < m) {
            h.pop();
            h.push(key);
            true
        } else {
            false
        }
    }

    /// The largest key a split keeps now (every key while it holds fewer than its cap).
    pub fn threshold(&self, split: &str) -> u64 {
        let cap = self.caps.get(split).copied().unwrap_or(0);
        match self.heaps.get(split) {
            Some(h) if h.len() >= cap => h.peek().copied().unwrap_or(u64::MAX),
            _ if cap == 0 => 0,
            _ => u64::MAX,
        }
    }
}

/// The counts of one shard (or of every shard, summed).
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Counts {
    pub sessions: usize,
    pub absent_from_splits: usize,
    pub silent_channel: usize,
    pub bleed: usize,
    pub kept_sessions: usize,
    /// Pauses with at least one ask, in every session decoded.
    pub candidates: usize,
    /// Candidates of the sessions dropped by the bleed filter or a silent channel.
    pub in_dropped_sessions: usize,
    pub backchannel: usize,
    pub overlap: usize,
    pub no_follow: usize,
    pub tie: usize,
    pub redaction: usize,
    /// Eligible records by split: `[hold, eot]`.
    pub eligible: BTreeMap<String, [usize; 2]>,
    /// Records written to the store at the time (before the caps' later evictions).
    pub written: BTreeMap<String, usize>,
    /// Seconds of audio decoded.
    pub decoded_seconds: f64,
}

impl Counts {
    fn add(&mut self, o: &Counts) {
        self.sessions += o.sessions;
        self.absent_from_splits += o.absent_from_splits;
        self.silent_channel += o.silent_channel;
        self.bleed += o.bleed;
        self.kept_sessions += o.kept_sessions;
        self.candidates += o.candidates;
        self.in_dropped_sessions += o.in_dropped_sessions;
        self.backchannel += o.backchannel;
        self.overlap += o.overlap;
        self.no_follow += o.no_follow;
        self.tie += o.tie;
        self.redaction += o.redaction;
        for (k, v) in &o.eligible {
            let e = self.eligible.entry(k.clone()).or_default();
            e[0] += v[0];
            e[1] += v[1];
        }
        for (k, v) in &o.written {
            *self.written.entry(k.clone()).or_default() += v;
        }
        self.decoded_seconds += o.decoded_seconds;
    }
}

/// The files of the conversation folder (`<data>/conv/`).
pub struct ConvDir {
    pub dir: PathBuf,
}

impl ConvDir {
    pub fn new(data: &Path) -> ConvDir {
        ConvDir {
            dir: data.join("conv"),
        }
    }
    pub fn store(&self) -> PathBuf {
        self.dir.join("store")
    }
    pub fn report(&self, tag: &str) -> PathBuf {
        self.dir.join(format!("conv-{tag}.json"))
    }
    pub fn sessions(&self, tag: &str) -> PathBuf {
        self.dir.join(format!("sessions-{tag}.jsonl"))
    }
    pub fn tails(&self, tag: &str) -> (PathBuf, PathBuf) {
        (
            self.dir.join(format!("tails-{tag}.bin")),
            self.dir.join(format!("tails-{tag}.jsonl")),
        )
    }
    pub fn xcheck(&self, tag: &str) -> PathBuf {
        self.dir.join(format!("xcheck-{tag}.jsonl"))
    }
    pub fn wav_dir(&self, tag: &str) -> PathBuf {
        self.dir.join(format!("wav-{tag}"))
    }
    pub fn parity(&self) -> PathBuf {
        self.dir.join("parity")
    }
}

/// The keys of every record a store holds, by split.
fn stored_keys(store: &Path, skip_tag: Option<&str>) -> Result<Vec<(String, u64)>> {
    let mut out = Vec::new();
    if !store.exists() {
        return Ok(out);
    }
    for e in fs::read_dir(store)? {
        let name = e?.file_name().to_string_lossy().into_owned();
        let Some(tag) = name
            .strip_prefix("manifest-")
            .and_then(|r| r.strip_suffix(".jsonl"))
        else {
            continue;
        };
        if Some(tag) == skip_tag {
            continue;
        }
        for line in BufReader::new(File::open(store.join(&name))?).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let v: Value = serde_json::from_str(&line)?;
            if let (Some(s), Some(k)) = (v["split"].as_str(), v["key"].as_u64()) {
                out.push((s.to_string(), k));
            }
        }
    }
    Ok(out)
}

/// Writes the records of one shard: the part, the transient tails, the cross-check inputs, the
/// parity windows, the sessions and the counts.
pub struct ShardWriter<'a> {
    cd: &'a ConvDir,
    tag: String,
    threads: usize,
    part: PartWriter,
    tails: BufWriter<File>,
    tail_lines: Vec<String>,
    xcheck: Vec<String>,
    sessions: Vec<String>,
    caps: Caps,
    /// The parity windows: (key, id), the smallest keys of the test records seen.
    parity: Vec<(u64, String)>,
    pub counts: Counts,
}

impl<'a> ShardWriter<'a> {
    /// Opens a shard's outputs; the caps start from the records already stored by other shards.
    pub fn new(cd: &'a ConvDir, tag: &str, threads: usize) -> Result<Self> {
        fs::create_dir_all(cd.store())?;
        let mut caps = Caps::new(&CAPS);
        for (s, k) in stored_keys(&cd.store(), Some(tag))? {
            caps.insert(&s, k);
        }
        let parity = read_parity(cd)?;
        let (tb, _) = cd.tails(tag);
        Ok(ShardWriter {
            cd,
            tag: tag.to_string(),
            threads,
            part: PartWriter::new(&cd.store(), tag)?,
            tails: BufWriter::new(File::create(tb)?),
            tail_lines: Vec::new(),
            xcheck: Vec::new(),
            sessions: Vec::new(),
            caps,
            parity,
            counts: Counts::default(),
        })
    }

    /// A session absent from `splits.json`: counted, not decoded.
    pub fn absent(&mut self, meta: &SessionMeta) {
        self.counts.sessions += 1;
        self.counts.absent_from_splits += 1;
        self.sessions.push(
            json!({"session": meta.id, "split": null, "drop": "absent_from_splits"}).to_string(),
        );
    }

    /// One decoded session (16 kHz channels) of split `split`.
    pub fn session(
        &mut self,
        meta: &SessionMeta,
        split: &str,
        channels: [Vec<f32>; 2],
    ) -> Result<()> {
        let sr = SAMPLE_RATE as f64;
        let red: Vec<(u64, u64)> = meta
            .redactions
            .iter()
            .map(|&(s, e)| ((s * sr).floor() as u64, (e * sr).ceil() as u64))
            .collect();
        let p = process_session(channels, &red);
        let c = &mut self.counts;
        c.sessions += 1;
        c.decoded_seconds += p.n as f64 / sr;
        c.candidates += p.candidates.len();
        let mut eligible = Vec::new();
        match p.drop {
            Some(d) => {
                c.in_dropped_sessions += p.candidates.len();
                match d {
                    SessionDrop::SilentChannel => c.silent_channel += 1,
                    SessionDrop::Bleed => c.bleed += 1,
                }
            }
            None => {
                c.kept_sessions += 1;
                for cand in &p.candidates {
                    match cand.outcome {
                        Err(Drop::Backchannel) => c.backchannel += 1,
                        Err(Drop::Overlap) => c.overlap += 1,
                        Err(Drop::NoFollow) => c.no_follow += 1,
                        Err(Drop::Tie) => c.tie += 1,
                        Err(Drop::Redaction) => c.redaction += 1,
                        Ok(l) => {
                            c.eligible.entry(split.to_string()).or_default()[l.eot as usize] += 1;
                            eligible.push((cand, l));
                        }
                    }
                }
            }
        }
        self.sessions.push(
            json!({"session": meta.id, "split": split, "samples": p.n,
                "speakers": meta.speakers, "gains": p.gains, "bleed": p.bleed, "drop": p.drop,
                "candidates": p.candidates.len(), "eligible": eligible.len()})
            .to_string(),
        );
        if p.drop.is_some() {
            return Ok(());
        }
        // the cross-check's inputs: every eligible record of a validation session, before the caps
        if split == "val" {
            let wd = self.cd.wav_dir(&self.tag);
            fs::create_dir_all(&wd)?;
            let mut files = Vec::new();
            for (ch, x) in p.pcm.iter().enumerate() {
                let f = wd.join(format!("{}.{ch}.wav", meta.id));
                fs::write(&f, write_wav_f32(x))?;
                files.push(f.file_name().unwrap().to_string_lossy().into_owned());
            }
            let energy: Vec<Value> = eligible
                .iter()
                .map(|(cand, l)| json!({"channel": cand.channel, "o": cand.pause.offset, "eot": l.eot}))
                .collect();
            self.xcheck.push(
                json!({"session": meta.id, "samples": p.n, "redactions": red, "wav": files,
                    "energy": energy})
                .to_string(),
            );
        }
        // the records the caps keep now
        let mut keep = Vec::new();
        for (cand, l) in &eligible {
            let id = format!("{}:{}:{}", meta.id, cand.channel, cand.pause.offset);
            let key = record_key(&id);
            if self.caps.insert(split, key) {
                let o = cand.pause.offset;
                let dmax = *cand.pause.asks.last().expect("a pause has an ask");
                let (_, _, cut_hops) = record_hops(o, dmax);
                let rel = |t: Option<u64>| t.map(|t| (t as f64 - o as f64) / sr);
                keep.push(Record {
                    id,
                    session: meta.id.clone(),
                    channel: cand.channel,
                    o_sample: o,
                    o: o as f64 / sr,
                    label: l.eot,
                    dmax_ms: dmax,
                    tb: rel(l.tb),
                    r: rel(l.resume),
                    split: split.to_string(),
                    speaker: meta.speakers[cand.channel].clone(),
                    cut_hops,
                    key,
                });
            }
        }
        let pcm = &p.pcm;
        let hops = par_map(&keep, self.threads, |fe, r| {
            let (first, last, _) = record_hops(r.o_sample, r.dmax_ms);
            hops_range(fe, &pcm[r.channel], first, last)
        });
        for (r, h) in keep.iter().zip(hops) {
            let n = h.len() / TURN_CHANNELS;
            self.part.put(&h, n, serde_json::to_value(r)?)?;
            *self.counts.written.entry(split.to_string()).or_default() += 1;
            if split == "val" || split == "test" {
                let tail = left_padded(
                    &pcm[r.channel],
                    (r.o_sample + FIRST_ASK_MS * MS) as usize,
                    TAIL_SAMPLES,
                );
                for x in &tail {
                    self.tails.write_all(&x.to_le_bytes())?;
                }
                self.tail_lines.push(
                    json!({"id": r.id, "index": self.tail_lines.len(), "split": split}).to_string(),
                );
            }
            if split == "test" {
                self.parity_offer(r, &pcm[r.channel])?;
            }
        }
        Ok(())
    }

    /// Keeps the 16-bit WAV of the 4 s window at d = 0.2 s of the test records of the smallest
    /// keys (§7's WASM parity; local only, deleted with the data).
    fn parity_offer(&mut self, r: &Record, pcm: &[f32]) -> Result<()> {
        // a shard redone after a crash offers its records again
        if self.parity.iter().any(|(_, id)| *id == r.id) {
            return Ok(());
        }
        if self.parity.len() >= PARITY_RECORDS
            && self.parity.last().is_some_and(|&(k, _)| r.key >= k)
        {
            return Ok(());
        }
        let dir = self.cd.parity();
        fs::create_dir_all(&dir)?;
        let end = (r.o_sample + FIRST_ASK_MS * MS) as usize;
        let last = (end - TURN_SPAN) / HOP;
        let start = (last + 1).saturating_sub(sky_infer::audio::TURN_WINDOW_HOPS) * HOP;
        let file = format!("{}.wav", r.id.replace(':', "_"));
        fs::write(
            dir.join(&file),
            dsp::write_wav(&pcm[start..end], SAMPLE_RATE),
        )?;
        self.parity.push((r.key, r.id.clone()));
        self.parity.sort();
        while self.parity.len() > PARITY_RECORDS {
            let (_, id) = self.parity.pop().expect("non-empty");
            fs::remove_file(dir.join(format!("{}.wav", id.replace(':', "_")))).ok();
        }
        let lines: String = self
            .parity
            .iter()
            .map(|(k, id)| {
                format!(
                    "{}\n",
                    json!({"id": id, "key": k, "split": "test", "wav": format!("{}.wav", id.replace(':', "_"))})
                )
            })
            .collect();
        fs::write(dir.join("records.jsonl"), lines)?;
        Ok(())
    }

    /// Finishes the shard: the part, the tails, the cross-check inputs, the sessions, then the
    /// counts (the shard's marker, written last).
    pub fn finish(mut self) -> Result<Counts> {
        self.part.finish()?;
        self.tails.flush()?;
        let (_, tl) = self.cd.tails(&self.tag);
        write_lines(&tl, &self.tail_lines)?;
        if !self.xcheck.is_empty() {
            write_lines(&self.cd.xcheck(&self.tag), &self.xcheck)?;
        }
        write_lines(&self.cd.sessions(&self.tag), &self.sessions)?;
        if self.tail_lines.is_empty() {
            let (tb, _) = self.cd.tails(&self.tag);
            fs::remove_file(tb).ok();
            fs::remove_file(tl).ok();
        }
        fs::write(
            self.cd.report(&self.tag),
            serde_json::to_string_pretty(&self.counts)? + "\n",
        )?;
        Ok(self.counts)
    }
}

fn write_lines(path: &Path, lines: &[String]) -> Result<()> {
    let tmp = path.with_extension("part");
    let mut f = BufWriter::new(File::create(&tmp)?);
    for l in lines {
        writeln!(f, "{l}")?;
    }
    f.flush()?;
    drop(f);
    fs::rename(&tmp, path)?;
    Ok(())
}

fn read_parity(cd: &ConvDir) -> Result<Vec<(u64, String)>> {
    let p = cd.parity().join("records.jsonl");
    if !p.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for line in BufReader::new(File::open(p)?).lines() {
        let v: Value = serde_json::from_str(&line?)?;
        if let (Some(k), Some(id)) = (v["key"].as_u64(), v["id"].as_str()) {
            out.push((k, id.to_string()));
        }
    }
    out.sort();
    Ok(out)
}

// ----------------------------------------------------------------------------- compaction

/// Finishes a compaction a crash interrupted: a part whose new manifest is written but not yet
/// in place.
fn recover(store: &Path) -> Result<()> {
    if !store.exists() {
        return Ok(());
    }
    for e in fs::read_dir(store)? {
        let name = e?.file_name().to_string_lossy().into_owned();
        let Some(tag) = name
            .strip_prefix("manifest-")
            .and_then(|r| r.strip_suffix(".jsonl.compact"))
        else {
            continue;
        };
        let feat_tmp = store.join(format!("{}.compact", feat_name(tag)));
        if feat_tmp.exists() {
            fs::rename(&feat_tmp, store.join(feat_name(tag)))?;
        }
        fs::rename(store.join(&name), store.join(manifest_name(tag)))?;
    }
    Ok(())
}

/// Removes from the parts the records the caps no longer keep: every part with an evicted
/// record when `full`, else only the parts that lost at least a quarter of their records.
/// Returns the records removed.
pub fn compact(store: &Path, caps: &Caps, full: bool) -> Result<usize> {
    recover(store)?;
    let mut removed = 0;
    let mut tags: Vec<String> = fs::read_dir(store)?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let n = e.file_name().to_string_lossy().into_owned();
            n.strip_prefix("manifest-")
                .and_then(|r| r.strip_suffix(".jsonl"))
                .map(str::to_string)
        })
        .collect();
    tags.sort();
    for tag in tags {
        let man = store.join(manifest_name(&tag));
        let lines: Vec<Value> = BufReader::new(File::open(&man)?)
            .lines()
            .map(|l| -> Result<Value> { Ok(serde_json::from_str(&l?)?) })
            .collect::<Result<_>>()?;
        let live = |v: &Value| {
            v["key"]
                .as_u64()
                .is_some_and(|k| k <= caps.threshold(v["split"].as_str().unwrap_or("")))
        };
        let gone = lines.iter().filter(|v| !live(v)).count();
        if gone == 0 || (!full && gone * 4 < lines.len()) {
            continue;
        }
        let feat = fs::read(store.join(feat_name(&tag)))?;
        let feat_tmp = store.join(format!("{}.compact", feat_name(&tag)));
        let man_tmp = store.join(format!("{}.compact", manifest_name(&tag)));
        let mut out = BufWriter::new(File::create(&feat_tmp)?);
        let mut m = BufWriter::new(File::create(&man_tmp)?);
        let mut offset = 0u64;
        for v in lines.iter().filter(|v| live(v)) {
            let (o, n) = (
                v["offset"].as_u64().unwrap_or(0) as usize,
                v["hops"].as_u64().unwrap_or(0) as usize,
            );
            out.write_all(&feat[o..o + n * HOP_BYTES])?;
            let mut v = v.clone();
            v["offset"] = json!(offset);
            offset += (n * HOP_BYTES) as u64;
            writeln!(m, "{v}")?;
        }
        out.flush()?;
        m.flush()?;
        drop((out, m));
        // the part disappears from the store while its files are swapped, then comes back
        fs::remove_file(&man)?;
        fs::rename(&feat_tmp, store.join(feat_name(&tag)))?;
        fs::rename(&man_tmp, &man)?;
        removed += gone;
    }
    Ok(removed)
}

// ----------------------------------------------------------------------------- fetch-conv

/// Options of `fetch-conv`.
#[derive(Clone, Debug)]
pub struct ConvOptions {
    /// The data folder: `shards/` (transient), `conv/`, `hf/`.
    pub out: PathBuf,
    /// Shards to process (all when empty).
    pub shards: Vec<usize>,
    /// Sessions read per shard (0: all; a dry run).
    pub max_sessions: usize,
    pub threads: usize,
    /// Only make the store the caps' exact draw and write `CONV.json`.
    pub finalize: bool,
    pub guard: Guard,
}

/// DualTurn's `splits.json`: session → split.
pub fn read_splits(bytes: &[u8]) -> Result<HashMap<String, String>> {
    let v: Value = serde_json::from_slice(bytes)?;
    let m = v["splits"]
        .as_object()
        .ok_or_else(|| anyhow!("splits.json: no `splits` object"))?;
    m.iter()
        .map(|(k, s)| {
            let s = s.as_str().ok_or_else(|| anyhow!("splits.json: {k}"))?;
            if !["train", "val", "test"].contains(&s) {
                bail!("splits.json: {k}: split {s:?}");
            }
            Ok((k.clone(), s.to_string()))
        })
        .collect()
}

fn splits(hf: &Hf, out: &Path, guard: &Guard) -> Result<HashMap<String, String>> {
    let files = hf.repo_files(RepoKind::Dataset, SPLITS_REPO, SPLITS_REVISION)?;
    let f = files
        .iter()
        .find(|f| f.path == "splits.json")
        .ok_or_else(|| anyhow!("{SPLITS_REPO}: no splits.json"))?;
    let dest = out.join("hf").join("dualturn-splits.json");
    download_guarded(hf, SPLITS_REPO, SPLITS_REVISION, f, &dest, guard, 0)?;
    let bytes = fs::read(&dest)?;
    let h = crate::turn::sha256_hex(&bytes);
    if h != SPLITS_SHA256 {
        bail!(
            "{}: sha256 {h}, the pre-registration's is {SPLITS_SHA256}",
            dest.display()
        );
    }
    read_splits(&bytes)
}

fn shard_files(hf: &Hf) -> Result<Vec<RepoFile>> {
    let mut files: Vec<RepoFile> = hf
        .repo_files(RepoKind::Dataset, OTO_REPO, OTO_REVISION)?
        .into_iter()
        .filter(|f| f.path.starts_with("data/train/shard-") && f.path.ends_with(".tar"))
        .collect();
    files.sort_by(|a, b| a.path.cmp(&b.path));
    if files.len() != OTO_SHARDS {
        bail!(
            "{OTO_REPO}@{OTO_REVISION}: {} shards, the pre-registration has {OTO_SHARDS}",
            files.len()
        );
    }
    Ok(files)
}

/// One shard's tar through the shard writer (`max_sessions` 0: all).
pub fn shard_records(
    tar: &Path,
    w: &mut ShardWriter,
    split_of: &HashMap<String, String>,
    max_sessions: usize,
) -> Result<()> {
    // the flac and json members of each session key, until both are read
    type Pair = (Option<Vec<u8>>, Option<Vec<u8>>);
    let mut pending: HashMap<String, Pair> = HashMap::new();
    let mut done = 0usize;
    for_each_tar_file(File::open(tar)?, false, |path, bytes| {
        let name = path.rsplit('/').next().unwrap_or(path);
        let Some((key, ext)) = name.rsplit_once('.') else {
            return Ok(true);
        };
        let e = pending.entry(key.to_string()).or_default();
        match ext {
            "flac" => e.0 = Some(bytes),
            "json" => e.1 = Some(bytes),
            _ => return Ok(true),
        }
        if e.0.is_some() && e.1.is_some() {
            let (flac, js) = pending.remove(key).expect("just seen");
            let meta = parse_meta(&js.expect("json"))?;
            if meta.id != key {
                bail!("{}: member {key} holds session {}", tar.display(), meta.id);
            }
            match split_of.get(&meta.id) {
                None => w.absent(&meta),
                Some(split) => {
                    let (chs, rate) = decode_flac_channels(&flac.expect("flac"))
                        .with_context(|| format!("decoding {}", meta.id))?;
                    if chs.len() != 2 {
                        bail!("{}: {} channels, two expected", meta.id, chs.len());
                    }
                    let mut it = chs
                        .into_iter()
                        .map(|c| dsp::resample(&c, rate, SAMPLE_RATE));
                    let chans = [it.next().expect("two"), it.next().expect("two")];
                    w.session(&meta, split, chans)?;
                }
            }
            done += 1;
        }
        Ok(max_sessions == 0 || done < max_sessions)
    })?;
    if max_sessions == 0 && !pending.is_empty() {
        bail!(
            "{}: {} members without their pair",
            tar.display(),
            pending.len()
        );
    }
    Ok(())
}

/// Removes what an interrupted run of a shard left (it is redone from its download).
fn clear_shard(cd: &ConvDir, tag: &str) -> Result<()> {
    let store = cd.store();
    for f in [
        store.join(manifest_name(tag)),
        store.join(feat_name(tag)),
        cd.tails(tag).0,
        cd.tails(tag).1,
        cd.xcheck(tag),
        cd.sessions(tag),
    ] {
        fs::remove_file(f).ok();
    }
    fs::remove_dir_all(cd.wav_dir(tag)).ok();
    Ok(())
}

/// `fetch-conv`.
pub fn fetch_conv(hf: &Hf, o: &ConvOptions) -> Result<Value> {
    let cd = ConvDir::new(&o.out);
    fs::create_dir_all(cd.store())?;
    recover(&cd.store())?;
    if o.finalize {
        return finalize(&cd);
    }
    let split_of = splits(hf, &o.out, &o.guard)?;
    let files = shard_files(hf)?;
    let shards: Vec<usize> = if o.shards.is_empty() {
        (0..files.len()).collect()
    } else {
        o.shards.clone()
    };
    for &i in &shards {
        let file = files
            .get(i)
            .ok_or_else(|| anyhow!("shard {i}: the corpus has {} shards", files.len()))?;
        let tag = shard_tag(i);
        let tar = o.out.join("shards").join(format!("conv-{tag}.tar"));
        if cd.report(&tag).exists() {
            fs::remove_file(&tar).ok();
            continue;
        }
        clear_shard(&cd, &tag)?;
        download_guarded(
            hf,
            OTO_REPO,
            OTO_REVISION,
            file,
            &tar,
            &o.guard,
            SHARD_RESERVE,
        )?;
        o.guard
            .check(SHARD_RESERVE, &format!("records of shard {tag}"))?;
        let t0 = std::time::Instant::now();
        let mut w = ShardWriter::new(&cd, &tag, o.threads)?;
        shard_records(&tar, &mut w, &split_of, o.max_sessions)?;
        let c = w.finish()?;
        fs::remove_file(&tar)?;
        // the parts that lost a quarter of their records to the caps since they were written
        let mut caps = Caps::new(&CAPS);
        for (s, k) in stored_keys(&cd.store(), None)? {
            caps.insert(&s, k);
        }
        let removed = compact(&cd.store(), &caps, false)?;
        eprintln!(
            "[fetch-conv] shard {tag}: {} sessions ({} kept), {} candidates, eligible {:?}, written {:?}, {removed} compacted out, {:.0} s",
            c.sessions,
            c.kept_sessions,
            c.candidates,
            c.eligible,
            c.written,
            t0.elapsed().as_secs_f64()
        );
    }
    summary(&cd, false)
}

/// The store made the caps' exact draw, and `CONV.json` (§3.2 counts, §3.3's speaker-unseen
/// subset).
pub fn finalize(cd: &ConvDir) -> Result<Value> {
    let mut caps = Caps::new(&CAPS);
    for (s, k) in stored_keys(&cd.store(), None)? {
        caps.insert(&s, k);
    }
    let removed = compact(&cd.store(), &caps, true)?;
    eprintln!("[fetch-conv] finalize: {removed} records compacted out");
    summary(cd, true)
}

/// Sums the counts of every finished shard into `CONV.json`; with `last`, also the stored records
/// by split and class, the asks' distribution and the speaker-unseen subset of the test records.
pub fn summary(cd: &ConvDir, last: bool) -> Result<Value> {
    let mut total = Counts::default();
    let mut done = Vec::new();
    for i in 0..OTO_SHARDS {
        let tag = shard_tag(i);
        let p = cd.report(&tag);
        if p.exists() {
            let c: Counts = serde_json::from_str(&fs::read_to_string(&p)?)?;
            total.add(&c);
            done.push(i);
        }
    }
    let mut v = json!({
        "repo": OTO_REPO, "revision": OTO_REVISION, "splits": {"repo": SPLITS_REPO,
        "revision": SPLITS_REVISION, "sha256": SPLITS_SHA256},
        "shards_done": done.len(), "shards": OTO_SHARDS, "counts": total,
        "caps": CAPS.iter().map(|(s, c)| (s.to_string(), *c)).collect::<BTreeMap<_, _>>(),
        "cap_seed": CAP_SEED,
        "rule": "docs/s6-ear3-prereg.md §3.2 (amendment 1): drops counted by the first rule that applies, in the order of item 5; sessions absent from splits.json are not decoded",
    });
    if last {
        let mut train_speakers = HashSet::new();
        for i in &done {
            let p = cd.sessions(&shard_tag(*i));
            if !p.exists() {
                continue;
            }
            for line in BufReader::new(File::open(p)?).lines() {
                let s: Value = serde_json::from_str(&line?)?;
                if s["split"] == "train" {
                    for sp in s["speakers"].as_array().into_iter().flatten() {
                        if let Some(x) = sp.as_str().filter(|x| !x.is_empty()) {
                            train_speakers.insert(x.to_string());
                        }
                    }
                }
            }
        }
        let st = sky_infer::turn::store::Store::open(&cd.store())?;
        let mut stored: BTreeMap<String, [usize; 2]> = BTreeMap::new();
        let mut dmax: BTreeMap<String, BTreeMap<u64, usize>> = BTreeMap::new();
        let (mut unseen, mut unseen_class) = (0usize, [0usize; 2]);
        for c in &st.clips {
            let split = c.split().to_string();
            let eot = c.label().unwrap_or(false) as usize;
            stored.entry(split.clone()).or_default()[eot] += 1;
            *dmax
                .entry(split.clone())
                .or_default()
                .entry(c.meta["dmax_ms"].as_u64().unwrap_or(0))
                .or_default() += 1;
            if split == "test" && !train_speakers.contains(c.meta["speaker"].as_str().unwrap_or(""))
            {
                unseen += 1;
                unseen_class[eot] += 1;
            }
        }
        v["stored"] = json!(stored);
        v["dmax_ms"] = json!(dmax);
        v["speaker_unseen_test"] = json!({"records": unseen, "hold_eot": unseen_class,
            "training_speakers": train_speakers.len()});
        v["final"] = json!(done.len() == OTO_SHARDS);
    }
    fs::write(
        cd.dir.join("CONV.json"),
        serde_json::to_string_pretty(&v)? + "\n",
    )?;
    Ok(v)
}

// ----------------------------------------------------------------------------- conv-crosscheck

/// Silero's offsets of a channel (§3.7): the end of each cleaned run that is followed by at least
/// 200 ms of silence (to the next run, or to the end), with the next run's start.
pub fn silero_offsets(a: &Channel) -> Vec<(u64, Option<u64>)> {
    let mut out = Vec::new();
    for (i, r) in a.runs.iter().enumerate() {
        let next = a.runs.get(i + 1).map(|x| x.start);
        let gap = next.unwrap_or(a.n).saturating_sub(r.end);
        if r.len() >= CLEAN_MS * MS && gap >= SILERO_GAP_MS * MS {
            out.push((r.end, next));
        }
    }
    out
}

/// The one-to-one matching of §3.7: every pair within `tol` samples, the closest first (ties by
/// position), each offset used once. Returns, for each `a`, the index of its match in `b`.
pub fn match_offsets(a: &[u64], b: &[u64], tol: u64) -> Vec<Option<usize>> {
    let mut pairs = Vec::new();
    for (i, &x) in a.iter().enumerate() {
        let lo = b.partition_point(|&y| y + tol < x);
        for (j, &y) in b.iter().enumerate().skip(lo) {
            if y > x + tol {
                break;
            }
            pairs.push((x.abs_diff(y), i, j));
        }
    }
    pairs.sort();
    let (mut ma, mut used) = (vec![None; a.len()], vec![false; b.len()]);
    for (_, i, j) in pairs {
        if ma[i].is_none() && !used[j] {
            ma[i] = Some(j);
            used[j] = true;
        }
    }
    ma
}

/// Wilson's 95% interval of `k` successes in `n`.
pub fn wilson(k: usize, n: usize) -> (f64, f64) {
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let (z, n_, p) = (1.959_963_984_540_054, n as f64, k as f64 / n as f64);
    let d = 1.0 + z * z / n_;
    let c = (p + z * z / (2.0 * n_)) / d;
    let h = z * ((p * (1.0 - p) + z * z / (4.0 * n_)) / n_).sqrt() / d;
    (c - h, c + h)
}

/// The tallies of the cross-check.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Crosscheck {
    pub sessions: usize,
    pub eligible: usize,
    /// Matched pairs with a label on both paths, and those whose labels agree.
    pub both_labelled: usize,
    pub agree: usize,
    /// Rows: the energy label (hold, eot); columns: Silero's.
    pub table: [[usize; 2]; 2],
    /// By the energy record's class (hold, eot).
    pub unmatched: [usize; 2],
    pub silero_dropped: [usize; 2],
    pub eligible_by_class: [usize; 2],
    /// Silero offset − energy offset of the matched pairs, in ms.
    pub diffs_ms: Vec<i64>,
    pub silero_offsets: usize,
    pub silero_drops: BTreeMap<String, usize>,
}

/// One validation session's cross-check: energy records `(channel, o, eot)` against the Silero
/// runs of its two channels.
pub fn crosscheck_session(
    energy: &[(usize, u64, bool)],
    silero_raw: [&[Run]; 2],
    n: u64,
    redactions: &[(u64, u64)],
    x: &mut Crosscheck,
) {
    let chans = [
        Channel::new(silero_raw[0], n),
        Channel::new(silero_raw[1], n),
    ];
    x.sessions += 1;
    for a in 0..2 {
        let offs = silero_offsets(&chans[a]);
        let outcomes: Vec<_> = offs
            .iter()
            .map(|&(o, r)| label(&chans[a], &chans[1 - a], o, r, redactions))
            .collect();
        x.silero_offsets += offs.len();
        for d in outcomes.iter().filter_map(|r| r.err()) {
            *x.silero_drops
                .entry(
                    serde_json::to_value(d)
                        .unwrap()
                        .as_str()
                        .unwrap_or("")
                        .to_string(),
                )
                .or_default() += 1;
        }
        let mut e: Vec<(u64, bool)> = energy
            .iter()
            .filter(|r| r.0 == a)
            .map(|r| (r.1, r.2))
            .collect();
        e.sort();
        let eo: Vec<u64> = e.iter().map(|r| r.0).collect();
        let so: Vec<u64> = offs.iter().map(|r| r.0).collect();
        for (i, m) in match_offsets(&eo, &so, MATCH_MS * MS)
            .into_iter()
            .enumerate()
        {
            let cls = e[i].1 as usize;
            x.eligible += 1;
            x.eligible_by_class[cls] += 1;
            match m {
                None => x.unmatched[cls] += 1,
                Some(j) => {
                    x.diffs_ms.push((so[j] as i64 - eo[i] as i64) / MS as i64);
                    match outcomes[j] {
                        Err(_) => x.silero_dropped[cls] += 1,
                        Ok(l) => {
                            x.both_labelled += 1;
                            x.table[cls][l.eot as usize] += 1;
                            x.agree += (l.eot as usize == cls) as usize;
                        }
                    }
                }
            }
        }
    }
}

impl Crosscheck {
    /// The report of §3.7, with the two bars.
    pub fn report(&self) -> Value {
        let agreement = self.agree as f64 / self.both_labelled.max(1) as f64;
        let (lo, hi) = wilson(self.agree, self.both_labelled);
        let unconf =
            self.unmatched.iter().sum::<usize>() + self.silero_dropped.iter().sum::<usize>();
        let unconfirmed = unconf as f64 / self.eligible.max(1) as f64;
        let mut d = self.diffs_ms.clone();
        d.sort();
        let q = |p: f64| -> Value {
            if d.is_empty() {
                Value::Null
            } else {
                json!(d[((p * (d.len() - 1) as f64).round() as usize).min(d.len() - 1)])
            }
        };
        let mut hist: BTreeMap<i64, usize> = BTreeMap::new();
        for &x in &d {
            *hist.entry(x.div_euclid(10) * 10).or_default() += 1;
        }
        let share = |a: [usize; 2]| -> Value {
            json!({"hold": a[0] as f64 / self.eligible_by_class[0].max(1) as f64,
                "eot": a[1] as f64 / self.eligible_by_class[1].max(1) as f64})
        };
        let pass_a = self.both_labelled > 0 && agreement >= AGREEMENT_BAR;
        let pass_u = self.eligible > 0 && unconfirmed <= UNCONFIRMED_BAR;
        json!({
            "about": "docs/s6-ear3-prereg.md §3.7: the labels of the validation sessions derived again from Silero VAD v6.2.3's speech runs with the same label code; energy records matched to Silero offsets within 150 ms, one to one, the closest first. Silero is used for this check only: no record is selected, dropped or relabelled by it.",
            "sessions": self.sessions, "eligible_energy_records": self.eligible,
            "eligible_by_class": {"hold": self.eligible_by_class[0], "eot": self.eligible_by_class[1]},
            "agreement": {"value": agreement, "wilson95": [lo, hi], "agree": self.agree,
                "pairs_labelled_by_both": self.both_labelled, "bar": AGREEMENT_BAR, "pass": pass_a},
            "unconfirmed": {"value": unconfirmed, "records": unconf, "bar": UNCONFIRMED_BAR, "pass": pass_u,
                "unmatched": {"hold": self.unmatched[0], "eot": self.unmatched[1]},
                "silero_dropped": {"hold": self.silero_dropped[0], "eot": self.silero_dropped[1]},
                "share_by_class": {"unmatched": share(self.unmatched), "silero_dropped": share(self.silero_dropped)}},
            "table": {"rows": "energy label (hold, eot)", "columns": "Silero label (hold, eot)", "counts": self.table},
            "time_difference_ms": {"silero_minus_energy": true, "n": d.len(), "p05": q(0.05), "p25": q(0.25),
                "median": q(0.5), "p75": q(0.75), "p95": q(0.95), "histogram_10ms": hist},
            "silero_offsets": self.silero_offsets, "silero_drops": self.silero_drops,
            "pass": pass_a && pass_u,
        })
    }
}

/// `conv-crosscheck`: every validation session's cross-check (`xcheck-*.jsonl` of `fetch-conv`,
/// `silero-*.jsonl` of `scripts/turn-vadcheck.py`) into `<data>/conv/CROSSCHECK.json`.
pub fn conv_crosscheck(data: &Path) -> Result<Value> {
    let cd = ConvDir::new(data);
    let mut x = Crosscheck::default();
    let mut shards = 0;
    let mut names: Vec<String> = fs::read_dir(&cd.dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("xcheck-") && n.ends_with(".jsonl"))
        .collect();
    names.sort();
    for name in names {
        let tag = &name["xcheck-".len()..name.len() - ".jsonl".len()];
        let sil = cd.dir.join(format!("silero-{tag}.jsonl"));
        let mut runs: HashMap<(String, usize), Vec<Run>> = HashMap::new();
        let f = File::open(&sil).with_context(|| {
            format!(
                "{} (scripts/turn-vadcheck.py has not run on shard {tag})",
                sil.display()
            )
        })?;
        for line in BufReader::new(f).lines() {
            let v: Value = serde_json::from_str(&line?)?;
            let key = (
                v["session"].as_str().unwrap_or("").to_string(),
                v["channel"].as_u64().unwrap_or(9) as usize,
            );
            let r = v["runs"]
                .as_array()
                .ok_or_else(|| anyhow!("{}: a line without runs", sil.display()))?
                .iter()
                .map(|p| Run {
                    start: p[0].as_u64().unwrap_or(0),
                    end: p[1].as_u64().unwrap_or(0),
                })
                .collect();
            runs.insert(key, r);
        }
        for line in BufReader::new(File::open(cd.dir.join(&name))?).lines() {
            let s: Value = serde_json::from_str(&line?)?;
            let id = s["session"].as_str().unwrap_or("").to_string();
            let red: Vec<(u64, u64)> = s["redactions"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|p| (p[0].as_u64().unwrap_or(0), p[1].as_u64().unwrap_or(0)))
                .collect();
            let energy: Vec<(usize, u64, bool)> = s["energy"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|r| {
                    (
                        r["channel"].as_u64().unwrap_or(0) as usize,
                        r["o"].as_u64().unwrap_or(0),
                        r["eot"].as_bool().unwrap_or(false),
                    )
                })
                .collect();
            let (a, b) = (
                runs.get(&(id.clone(), 0))
                    .ok_or_else(|| anyhow!("{}: no Silero runs of {id}:0", sil.display()))?,
                runs.get(&(id.clone(), 1))
                    .ok_or_else(|| anyhow!("{}: no Silero runs of {id}:1", sil.display()))?,
            );
            crosscheck_session(
                &energy,
                [a.as_slice(), b.as_slice()],
                s["samples"].as_u64().unwrap_or(0),
                &red,
                &mut x,
            );
        }
        shards += 1;
    }
    if shards == 0 {
        bail!(
            "{}: no cross-check input (xcheck-*.jsonl)",
            cd.dir.display()
        );
    }
    let mut v = x.report();
    v["shards"] = json!(shards);
    fs::write(
        cd.dir.join("CROSSCHECK.json"),
        serde_json::to_string_pretty(&v)? + "\n",
    )?;
    Ok(v)
}

// ----------------------------------------------------------------------------- the smoke fixture

/// A synthetic two-speaker session (tests and the `SMOKE=1` chain of `scripts/train-turn3.sh`):
/// steady two-partial tones over a background of uniform noise of amplitude `noise` on each
/// channel, at the given `(channel, start s, end s)` turns.
pub fn synth_session(
    secs: f64,
    turns: &[(usize, f64, f64)],
    seed: u64,
    noise: f32,
) -> [Vec<f32>; 2] {
    let n = (secs * SAMPLE_RATE as f64) as usize;
    let mut out = [vec![0.0f32; n], vec![0.0f32; n]];
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    for ch in out.iter_mut() {
        for x in ch.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *x = noise * (((s >> 40) as f64 / (1u64 << 24) as f64) - 0.5) as f32;
        }
    }
    for &(c, a, b) in turns {
        let (a, b) = ((a * 16_000.0) as usize, ((b * 16_000.0) as usize).min(n));
        // two partials at a steady level, with a slow glide of the lower one: every 10 ms frame
        // of a turn is speech to the energy VAD
        let f0 = 150.0 + 10.0 * (seed % 7) as f64;
        let mut ph = 0.0f64;
        for (k, x) in out[c][a..b].iter_mut().enumerate() {
            let t = k as f64 / 16_000.0;
            ph += (f0 + 20.0 * (t * 1.7).sin()) / 16_000.0;
            *x += (0.3 * (std::f64::consts::TAU * ph).sin()
                + 0.1 * (std::f64::consts::TAU * 3.1 * ph).sin()) as f32;
        }
    }
    out
}

/// `turn3-fixture`: a conversation store of synthetic sessions through the shard writer, with the
/// cross-check's inputs and Silero runs copied from the energy VAD (agreement 1 by construction).
/// For the `SMOKE=1` chain only: nothing of it is a result.
pub fn conv_fixture(data: &Path, sessions: usize) -> Result<Value> {
    let cd = ConvDir::new(data);
    let tag = "fixture";
    clear_shard(&cd, tag)?;
    let mut w = ShardWriter::new(&cd, tag, 2)?;
    let mut silero = Vec::new();
    for k in 0..sessions {
        let split = ["train", "train", "train", "val", "test"][k % 5];
        let mut turns = Vec::new();
        let mut t = 0.5;
        let mut who = 0usize;
        while t < 55.0 {
            let len = 1.2 + ((k * 7 + turns.len() * 3) % 9) as f64 * 0.25;
            turns.push((who, t, t + len));
            // a hold (the same speaker again) or a hand-over, after a pause of 0.4-1.6 s
            let hold = (k + turns.len()) % 3 == 0;
            t += len + 0.4 + ((k + turns.len()) % 5) as f64 * 0.3;
            if !hold {
                who = 1 - who;
            }
        }
        // no background: the bleed probe of the fixture then reads ties (AUC 0.5), not noise
        let chans = synth_session(60.0, &turns, k as u64 + 1, 0.0);
        let meta = SessionMeta {
            id: format!("fixture{k:03}"),
            duration: 60.0,
            redactions: Vec::new(),
            speakers: [format!("spk{}", k % 4), format!("spk{}", (k + 1) % 4)],
        };
        if split == "val" {
            let p = process_session(chans.clone(), &[]);
            for (ch, r) in p.raw_runs.iter().enumerate() {
                let runs: Vec<[u64; 2]> = r.iter().map(|x| [x.start, x.end]).collect();
                silero.push(json!({"session": meta.id, "channel": ch, "runs": runs}).to_string());
            }
        }
        w.session(&meta, split, chans)?;
    }
    let c = w.finish()?;
    write_lines(&cd.dir.join(format!("silero-{tag}.jsonl")), &silero)?;
    fs::remove_dir_all(cd.wav_dir(tag)).ok();
    let (tb, tl) = cd.tails(tag);
    fs::remove_file(tb).ok();
    fs::remove_file(tl).ok();
    let mut caps = Caps::new(&CAPS);
    for (s, k) in stored_keys(&cd.store(), None)? {
        caps.insert(&s, k);
    }
    compact(&cd.store(), &caps, true)?;
    Ok(json!({"fixture": tag, "counts": c}))
}

/// The `SMOKE=1` stand-in of `turn-extend`: for the first `max` clips of each split of a Smart
/// Turn store, an extension of 240 copies of the clip's last hop (not a result).
pub fn ext_fixture(st: &Path, out: &Path, max: usize) -> Result<Value> {
    let s = sky_infer::turn::store::Store::open(st)?;
    let mut part = PartWriter::new(out, "fixture")?;
    let mut taken: HashMap<String, usize> = HashMap::new();
    for c in &s.clips {
        let k = taken.entry(c.split().to_string()).or_default();
        if *k >= max || c.hops == 0 {
            continue;
        }
        *k += 1;
        let h = s.hops(c)?;
        let last = &h[(c.hops - 1) * TURN_CHANNELS..];
        let ext: Vec<f32> = (0..crate::turn::EXT_HOPS)
            .flat_map(|_| last.iter().copied())
            .collect();
        part.put(
            &ext,
            crate::turn::EXT_HOPS,
            json!({"id": c.id, "split": c.split(), "zero_fill": false, "last_hop_match": true, "fixture": true}),
        )?;
    }
    let n = part.finish()?;
    Ok(json!({"clips": n, "from": st}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sky_infer::turn::store::Store;

    const S: u64 = 16_000;

    /// A fresh temporary folder (removed by the test that made it).
    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("conv-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn chan(runs: &[(f64, f64)], secs: f64) -> Channel {
        let raw: Vec<Run> = runs
            .iter()
            .map(|&(a, b)| Run {
                start: (a * S as f64) as u64,
                end: (b * S as f64) as u64,
            })
            .collect();
        Channel::new(&raw, (secs * S as f64) as u64)
    }

    #[test]
    fn cleaning_removes_short_runs_then_fills_short_gaps() {
        let ms = |a: u64, b: u64| Run {
            start: a * 16,
            end: b * 16,
        };
        let raw = [
            ms(0, 100),
            ms(200, 500),
            ms(600, 900),
            ms(1000, 1100),
            ms(1200, 1350),
        ];
        // 0-100 and 1000-1100 are under 150 ms; the 100 ms gap 500-600 is filled; 1200-1350 is
        // exactly 150 ms and stays; its 300 ms gap from 900 is not filled
        assert_eq!(clean(&raw, 150 * 16), vec![ms(200, 900), ms(1200, 1350)]);
    }

    #[test]
    fn kinds_follow_dualturns_card() {
        let c = chan(
            &[(0.0, 3.0), (4.5, 5.2), (6.5, 7.5), (7.9, 8.4), (10.0, 10.5)],
            12.0,
        );
        assert_eq!(
            c.kinds,
            vec![
                Kind::Take,        // 3 s
                Kind::Backchannel, // 0.7 s, 1.5 s before, 1.3 s after
                Kind::Take,        // 1.0 s, but only 0.4 s of silence after: not a backchannel
                Kind::Other,       // 0.5 s, 0.4 s before
                Kind::Backchannel, // 0.5 s, 1.6 s before and 1.5 s to the end
            ]
        );
        assert_eq!(c.utterance_at(3 * S), Some(0));
        assert_eq!(c.utterance_at(4 * S), Some(0));
        assert_eq!(c.utterance_at(0), None);
    }

    #[test]
    fn the_label_rule_on_hand_made_channels() {
        let none: &[(u64, u64)] = &[];
        // A speaks 0-3 s; B takes the floor at 3.5 s for 2 s; A resumes at 6 s: end of turn
        let a = chan(&[(0.0, 3.0), (6.0, 8.0)], 20.0);
        let b = chan(&[(3.5, 5.5)], 20.0);
        let l = label(&a, &b, 3 * S, Some(6 * S), none).unwrap();
        assert!(l.eot);
        assert_eq!(l.tb, Some(3 * S + S / 2));
        // A resumes first, B only backchannels (0.5 s, isolated) during the pause: hold
        let a = chan(&[(0.0, 3.0), (5.0, 8.0)], 20.0);
        let b = chan(&[(3.6, 4.1), (9.5, 12.0)], 20.0);
        let l = label(&a, &b, 3 * S, Some(5 * S), none).unwrap();
        assert!(!l.eot && l.tb == Some(9 * S + S / 2));
        // nobody within 4 s: dropped
        let a = chan(&[(0.0, 3.0), (7.5, 9.0)], 20.0);
        let b = chan(&[(7.2, 9.0)], 20.0);
        assert_eq!(
            label(&a, &b, 3 * S, Some(7 * S + S / 2), none),
            Err(Drop::NoFollow)
        );
        // B under way at o: an overlap
        let a = chan(&[(0.0, 3.0), (5.0, 6.0)], 20.0);
        let b = chan(&[(2.0, 4.0)], 20.0);
        assert_eq!(label(&a, &b, 3 * S, Some(5 * S), none), Err(Drop::Overlap));
        // a floor take begun 0.5 s before o is an overlap too (a take lasts 1 s, so it is under
        // way at o: the rule's second half adds nothing to its first)
        let b = chan(&[(2.5, 3.8)], 20.0);
        assert_eq!(label(&a, &b, 3 * S, Some(5 * S), none), Err(Drop::Overlap));
        // one that ended 0.5 s before o is not
        let b = chan(&[(1.0, 2.5)], 20.0);
        assert!(label(&a, &b, 3 * S, Some(5 * S), none).is_ok());
        // A's own utterance is a backchannel (0.6 s, isolated): dropped
        let a = chan(&[(2.4, 3.0), (5.0, 7.0)], 20.0);
        let b = chan(&[(10.0, 12.0)], 20.0);
        assert_eq!(
            label(&a, &b, 3 * S, Some(5 * S), none),
            Err(Drop::Backchannel)
        );
        // a redaction 0.4 s after o + 4 s: dropped; 0.6 s after: kept
        let a = chan(&[(0.0, 3.0), (5.0, 7.0)], 20.0);
        let b = chan(&[(12.0, 14.0)], 20.0);
        assert_eq!(
            label(&a, &b, 3 * S, Some(5 * S), &[(7 * S + 4 * S / 10, 8 * S)]),
            Err(Drop::Redaction)
        );
        assert!(label(&a, &b, 3 * S, Some(5 * S), &[(7 * S + 6 * S / 10, 8 * S)]).is_ok());
        // the same sample for both: a tie
        let b = chan(&[(5.0, 7.0)], 20.0);
        assert_eq!(label(&a, &b, 3 * S, Some(5 * S), none), Err(Drop::Tie));
    }

    #[test]
    fn bleed_and_gain() {
        let rms = [0.01, 0.01, 0.05, 0.05, 0.05];
        let me = [false; 5];
        let other = [false, false, true, true, true];
        assert!((bleed_ratio(&rms, &me, &other).unwrap() - 5.0).abs() < 1e-12);
        assert_eq!(bleed_ratio(&rms, &me, &[false; 5]), None);
        assert_eq!(
            bleed_ratio(&[0.0; 4], &[false; 4], &[true, false, true, false]),
            Some(1.0)
        );
        let pcm: Vec<f32> = (0..16_000)
            .map(|i| if i < 8000 { 0.2 } else { 0.0 })
            .collect();
        let g = gain(&pcm).unwrap();
        assert!((g as f64 * 0.2 - 0.1).abs() < 1e-6, "{g}");
        assert_eq!(gain(&vec![0.0; 16_000]), None);
        assert_eq!(percentile(&[3.0, 1.0, 2.0, 4.0], 0.95), Some(4.0));
    }

    #[test]
    fn a_synthetic_session_has_the_known_answer() {
        // A 0.5-3.0 s; B 3.6-6.0 s (A's pause at 3.0 s: end of turn); A 6.4-8.0 s and 8.8-10.5 s
        // (A's pause at 8.0 s: hold, A resumes at 8.8 s); B 11.2-13.0 s (A's pause at 10.5 s: end
        // of turn); A 13.6-16 s, then nobody (no follow); B's pauses: at 6.0 s A takes the floor at
        // 6.4 s (end of turn); at 13.0 s A takes it at 13.6 s (end of turn)
        let turns = [
            (0, 0.5, 3.0),
            (1, 3.6, 6.0),
            (0, 6.4, 8.0),
            (0, 8.8, 10.5),
            (1, 11.2, 13.0),
            (0, 13.6, 16.0),
        ];
        let ch = synth_session(22.0, &turns, 5, 0.0005);
        let p = process_session(ch, &[]);
        assert_eq!(p.drop, None, "bleed {:?} gains {:?}", p.bleed, p.gains);
        let got: Vec<(usize, f64, std::result::Result<bool, Drop>)> = p
            .candidates
            .iter()
            .map(|c| {
                (
                    c.channel,
                    (c.pause.offset as f64 / 1600.0).round() / 10.0,
                    c.outcome.map(|l| l.eot),
                )
            })
            .collect();
        assert_eq!(
            got,
            vec![
                (0, 3.0, Ok(true)),
                (0, 8.0, Ok(false)),
                (0, 10.5, Ok(true)),
                (0, 16.0, Err(Drop::NoFollow)),
                (1, 6.0, Ok(true)),
                (1, 13.0, Ok(true)),
            ],
            "{:?}",
            p.raw_runs
        );
        // the asks of the hold: 0.2 to 0.8 s (A is silent over [8.0, 8.8) s and resumes at 8.8 s)
        assert_eq!(p.candidates[1].pause.asks, vec![200, 400, 600, 800]);
        // bleed: A's voice copied into B at a third of its level fails the filter
        let mut ch = synth_session(22.0, &turns, 5, 0.0005);
        let a0 = ch[0].clone();
        for (x, y) in ch[1].iter_mut().zip(&a0) {
            *x += 0.3 * y;
        }
        assert_eq!(process_session(ch, &[]).drop, Some(SessionDrop::Bleed));
    }

    #[test]
    fn record_hops_end_at_the_asks() {
        // o = 10 s: the last hop ending at or before o is (160000 - 640) / 160 = 996
        let (first, last, cut) = record_hops(160_000, 2400);
        assert_eq!((first, cut), (996 + 1 - 380, 380));
        assert_eq!(last, (160_000 + 38_400 - 640) / 160);
        assert_eq!(last - first + 1, 620);
        // near the session's start, fewer hops before the cut
        let (first, _, cut) = record_hops(2400 + 160 * 10, 200);
        assert_eq!(first, 0);
        assert_eq!(cut, (4000 - 640) / 160 + 1);
    }

    #[test]
    fn caps_keep_the_smallest_keys() {
        let mut c = Caps::new(&[("train", 3)]);
        for k in [50, 10, 40, 30, 20, 60] {
            c.insert("train", k);
        }
        assert_eq!(c.threshold("train"), 30);
        assert!(!c.insert("train", 35) && c.insert("train", 5));
        assert_eq!(c.threshold("train"), 20);
        assert_eq!(c.threshold("val"), 0);
    }

    #[test]
    fn matching_is_one_to_one_closest_first() {
        let ms = 16;
        // 1160 takes 1150 (10 ms) first; 1000 then takes 1100 (100 ms); 5000 has nothing within 150
        let a = [1000 * ms, 1160 * ms, 5000 * ms];
        let b = [1100 * ms, 1150 * ms, 5200 * ms];
        let m = match_offsets(&a, &b, 150 * ms);
        assert_eq!(m, vec![Some(0), Some(1), None]);
        let (lo, hi) = wilson(85, 100);
        assert!(
            lo < 0.85 && hi > 0.85 && lo > 0.76 && hi < 0.91,
            "{lo} {hi}"
        );
    }

    #[test]
    fn identical_runs_give_agreement_one_and_nothing_unconfirmed() {
        let turns = [
            (0, 0.5, 3.0),
            (1, 3.6, 6.0),
            (0, 6.4, 8.0),
            (0, 8.8, 10.5),
            (1, 11.2, 13.0),
            (0, 13.6, 16.0),
            (1, 16.5, 19.0),
        ];
        let p = process_session(synth_session(22.0, &turns, 9, 0.0005), &[]);
        let energy: Vec<(usize, u64, bool)> = p
            .candidates
            .iter()
            .filter_map(|c| c.outcome.ok().map(|l| (c.channel, c.pause.offset, l.eot)))
            .collect();
        assert!(energy.len() >= 5);
        let mut x = Crosscheck::default();
        crosscheck_session(
            &energy,
            [p.raw_runs[0].as_slice(), p.raw_runs[1].as_slice()],
            p.n,
            &[],
            &mut x,
        );
        let r = x.report();
        assert_eq!(r["agreement"]["value"], 1.0, "{r}");
        assert_eq!(r["unconfirmed"]["value"], 0.0, "{r}");
        assert_eq!(r["pass"], true);
        // shifted by 300 ms, nothing matches: every record is unconfirmed
        let shift: Vec<Vec<Run>> = p
            .raw_runs
            .iter()
            .map(|rs| {
                rs.iter()
                    .map(|r| Run {
                        start: r.start + 4800,
                        end: r.end + 4800,
                    })
                    .collect()
            })
            .collect();
        let mut y = Crosscheck::default();
        crosscheck_session(
            &energy,
            [shift[0].as_slice(), shift[1].as_slice()],
            p.n + 4800,
            &[],
            &mut y,
        );
        assert_eq!(y.report()["unconfirmed"]["value"], 1.0);
    }

    #[test]
    fn the_shard_writer_keeps_the_caps_and_compacts() {
        let dir = tmp("writer");
        let cd = ConvDir::new(&dir);
        let v = conv_fixture(&dir, 5).unwrap();
        let c: Counts = serde_json::from_value(v["counts"].clone()).unwrap();
        assert_eq!(c.sessions, 5);
        let st = Store::open(&cd.store()).unwrap();
        assert!(!st.clips.is_empty());
        for clip in &st.clips {
            let r: Record = serde_json::from_value(clip.meta.clone()).unwrap();
            assert_eq!(clip.hops, r.cut_hops + r.dmax_ms as usize / 10, "{}", r.id);
            assert!(r.cut_hops <= BACK_HOPS);
            assert_eq!(r.key, record_key(&r.id));
        }
        // the cross-check of the fixture agrees with itself
        let x = conv_crosscheck(&dir).unwrap();
        assert_eq!(x["agreement"]["value"], 1.0, "{x}");
        assert!(cd.parity().join("records.jsonl").exists());
        // compaction under a tighter cap keeps exactly the smallest keys, offsets consistent
        let keys: Vec<u64> = st
            .clips
            .iter()
            .filter(|c| c.split() == "train")
            .map(|c| c.meta["key"].as_u64().unwrap())
            .collect();
        let hops_before: HashMap<String, Vec<f32>> = st
            .clips
            .iter()
            .map(|c| (c.id.clone(), st.hops(c).unwrap()))
            .collect();
        let mut caps = Caps::new(&[("train", 3), ("val", 1_000_000), ("test", 1_000_000)]);
        for &k in &keys {
            caps.insert("train", k);
        }
        let removed = compact(&cd.store(), &caps, true).unwrap();
        assert_eq!(removed, keys.len() - 3);
        let st2 = Store::open(&cd.store()).unwrap();
        let mut sorted = keys.clone();
        sorted.sort();
        let kept: HashSet<u64> = st2
            .clips
            .iter()
            .filter(|c| c.split() == "train")
            .map(|c| c.meta["key"].as_u64().unwrap())
            .collect();
        assert_eq!(kept, sorted[..3].iter().copied().collect());
        for c in &st2.clips {
            assert_eq!(st2.hops(c).unwrap(), hops_before[&c.id]);
        }
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn metadata_reads_ids_redactions_and_speakers_only() {
        let j = br#"{"session_id": "abc", "duration": 12.5, "redacted_segments": [{"start_sec": 1.5, "end_sec": 2.0}],
            "speakers": [{"speaker_id": "s1", "channel": 1, "age": 30}, {"speaker_id": "s0", "channel": 0}], "surveys": []}"#;
        let m = parse_meta(j).unwrap();
        assert_eq!((m.id.as_str(), m.duration), ("abc", 12.5));
        assert_eq!(m.redactions, vec![(1.5, 2.0)]);
        assert_eq!(m.speakers, ["s0".to_string(), "s1".to_string()]);
        assert!(parse_meta(br#"{"session_id": "x", "redacted_segments": [[1, 2]]}"#).is_err());
        let s = read_splits(br#"{"seed": 42, "splits": {"a": "train", "b": "test"}}"#).unwrap();
        assert_eq!(s["b"], "test");
        assert!(read_splits(br#"{"splits": {"a": "dev"}}"#).is_err());
    }

    /// The parser on real bytes: one otoSpeech shard named by `SKY_OTO_SHARD` (and DualTurn's
    /// `splits.json` by `SKY_OTO_SPLITS`), through the shard writer into a temporary folder that
    /// is deleted. Run by hand: `cargo test -p sky-data conv::tests::one_real_shard -- --ignored`.
    #[test]
    #[ignore]
    fn one_real_shard() {
        let (Ok(tar), Ok(sp)) = (
            std::env::var("SKY_OTO_SHARD"),
            std::env::var("SKY_OTO_SPLITS"),
        ) else {
            panic!("set SKY_OTO_SHARD and SKY_OTO_SPLITS");
        };
        let split_of = read_splits(&fs::read(sp).unwrap()).unwrap();
        let dir = tmp("real");
        let cd = ConvDir::new(&dir);
        let mut w = ShardWriter::new(&cd, "00150", 2).unwrap();
        shard_records(Path::new(&tar), &mut w, &split_of, 1).unwrap();
        let c = w.finish().unwrap();
        // only the counts are printed: no record is kept
        eprintln!("{}", serde_json::to_string(&c).unwrap());
        assert_eq!(c.sessions, 1);
        assert!(c.candidates > 0);
        let st = Store::open(&cd.store()).unwrap();
        for clip in &st.clips {
            assert_eq!(
                clip.hops,
                clip.meta["cut_hops"].as_u64().unwrap() as usize
                    + clip.meta["dmax_ms"].as_u64().unwrap() as usize / 10
            );
        }
        fs::remove_dir_all(&dir).unwrap();
    }
}
