//! A measurement tool of s6-ear v3, separate from the stages (docs/s6-ear3-prereg.md, amendment
//! 1, the measurement on training data): the label rule of §3.2 under both detectors of §3.7 on
//! the **training** sessions of one otoSpeech shard, with two variants that are not adopted.
//!
//! * `conv-measure prepare --tar SHARD --splits splits.json --out DIR`: every training session of
//!   the shard (the others are skipped by their key, never decoded) through `process_session`;
//!   writes DIR/sessions.jsonl (the energy path's runs, pauses and drops, no audio) and, for
//!   `scripts/turn-vadcheck.py --tag meas`, DIR/xcheck-meas.jsonl and the gain-normalised
//!   channels DIR/wav-meas/.
//! * `conv-measure measure --dir DIR`: with Silero's runs (DIR/silero-meas.jsonl), the counts of
//!   each path, the runs' statistics per channel, §3.7's agreement, and the variants (i) silences
//!   under 300 ms filled in the runs that decide floor takes and backchannels, (ii) B's floor takes
//!   read from Silero's runs. Writes DIR/MEASURE.json; nothing else is kept.
//!
//! It calls the stages' own code (`process_session`, `label`, `silero_offsets`,
//! `crosscheck_session`) and changes none of it.

use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};
use sky_infer::audio::{self as dsp, SAMPLE_RATE};

use crate::audio::{decode_flac_channels, for_each_tar_file};
use crate::conv::{
    crosscheck_session, label, parse_meta, process_session, read_splits, silero_offsets,
    write_wav_f32, Channel, Crosscheck, Drop, Kind, Run, BACKCHANNEL_MS, CLEAN_MS, TAKE_MS,
};

const MS: u64 = 16;

fn runs_json(r: &[Run]) -> Value {
    json!(r.iter().map(|x| [x.start, x.end]).collect::<Vec<_>>())
}

fn runs_of_json(v: &Value) -> Vec<Run> {
    v.as_array()
        .into_iter()
        .flatten()
        .map(|p| Run {
            start: p[0].as_u64().unwrap_or(0),
            end: p[1].as_u64().unwrap_or(0),
        })
        .collect()
}

/// `prepare`: the training sessions of a shard.
pub fn prepare(tar: &Path, splits: &Path, out: &Path) -> Result<Value> {
    let split_of = read_splits(&fs::read(splits)?)?;
    let wav = out.join("wav-meas");
    fs::create_dir_all(&wav)?;
    let (mut sess, mut xc, mut skipped) = (Vec::new(), Vec::new(), BTreeMap::new());
    let mut flac: HashMap<String, Vec<u8>> = HashMap::new();
    for_each_tar_file(File::open(tar)?, false, |path, bytes| {
        let name = path.rsplit('/').next().unwrap_or(path);
        let Some((key, ext)) = name.rsplit_once('.') else {
            return Ok(true);
        };
        let split = split_of.get(key).map(String::as_str).unwrap_or("absent");
        if split != "train" {
            // a validation, test or unlisted session: skipped by its key, never decoded
            if ext == "flac" {
                *skipped.entry(split.to_string()).or_insert(0usize) += 1;
            }
            return Ok(true);
        }
        if ext == "flac" {
            flac.insert(key.to_string(), bytes);
            return Ok(true);
        }
        if ext != "json" {
            return Ok(true);
        }
        let meta = parse_meta(&bytes)?;
        let Some(f) = flac.remove(key) else {
            bail!("{key}: json before its flac");
        };
        let (chs, rate) = decode_flac_channels(&f)?;
        let mut it = chs
            .into_iter()
            .map(|c| dsp::resample(&c, rate, SAMPLE_RATE));
        let sr = SAMPLE_RATE as f64;
        let red: Vec<(u64, u64)> = meta
            .redactions
            .iter()
            .map(|&(s, e)| ((s * sr).floor() as u64, (e * sr).ceil() as u64))
            .collect();
        let p = process_session(
            [
                it.next().ok_or_else(|| anyhow!("no channel"))?,
                it.next().ok_or_else(|| anyhow!("one channel"))?,
            ],
            &red,
        );
        let mut files = Vec::new();
        for (ch, x) in p.pcm.iter().enumerate() {
            let n = format!("{}.{ch}.wav", meta.id);
            fs::write(wav.join(&n), write_wav_f32(x))?;
            files.push(n);
        }
        let pauses: Vec<Value> = p
            .candidates
            .iter()
            .map(|c| json!({"channel": c.channel, "o": c.pause.offset, "r": c.pause.resume}))
            .collect();
        sess.push(
            json!({"session": meta.id, "samples": p.n, "redactions": red, "drop": p.drop,
                "gains": p.gains, "bleed": p.bleed,
                "energy_runs": [runs_json(&p.raw_runs[0]), runs_json(&p.raw_runs[1])],
                "pauses": pauses})
            .to_string(),
        );
        xc.push(
            json!({"session": meta.id, "samples": p.n, "wav": files, "energy": []}).to_string(),
        );
        Ok(true)
    })?;
    fs::write(out.join("sessions.jsonl"), sess.join("\n") + "\n")?;
    fs::write(out.join("xcheck-meas.jsonl"), xc.join("\n") + "\n")?;
    Ok(json!({"training_sessions": sess.len(), "skipped_sessions": skipped}))
}

/// A channel built as `Channel::new` builds it, with the silences under `gap` filled (variant (i));
/// with `gap` = 150 ms it is `Channel::new`.
fn channel_with_gap(raw: &[Run], n: u64, gap: u64) -> Channel {
    let mut runs: Vec<Run> = Vec::new();
    for r in raw.iter().filter(|r| r.len() >= CLEAN_MS * MS) {
        match runs.last_mut() {
            Some(l) if r.start - l.end < gap => l.end = r.end,
            _ => runs.push(*r),
        }
    }
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

/// The counts of one path's outcomes.
#[derive(Default)]
struct Tally {
    candidates: usize,
    drops: BTreeMap<String, usize>,
    hold: usize,
    eot: usize,
}

impl Tally {
    fn add(&mut self, o: std::result::Result<crate::conv::Labelled, Drop>) {
        self.candidates += 1;
        match o {
            Ok(l) if l.eot => self.eot += 1,
            Ok(_) => self.hold += 1,
            Err(d) => {
                let k = serde_json::to_value(d)
                    .ok()
                    .and_then(|v| v.as_str().map(String::from));
                *self.drops.entry(k.unwrap_or_default()).or_default() += 1;
            }
        }
    }

    fn json(&self) -> Value {
        json!({"candidates": self.candidates, "drops": self.drops, "hold": self.hold, "eot": self.eot,
            "eot_share": self.eot as f64 / (self.eot + self.hold).max(1) as f64})
    }
}

/// The runs' statistics of one channel index, pooled over the sessions.
#[derive(Default)]
struct RunStats {
    lens_ms: Vec<u64>,
    takes: usize,
    backchannels: usize,
    speech: u64,
    samples: u64,
}

impl RunStats {
    fn add(&mut self, c: &Channel) {
        for (r, k) in c.runs.iter().zip(&c.kinds) {
            self.lens_ms.push(r.len() / MS);
            self.speech += r.len();
            self.takes += (*k == Kind::Take) as usize;
            self.backchannels += (*k == Kind::Backchannel) as usize;
        }
        self.samples += c.n;
    }

    fn json(&mut self) -> Value {
        self.lens_ms.sort();
        let q = |p: f64| -> Value {
            if self.lens_ms.is_empty() {
                Value::Null
            } else {
                let k =
                    ((p * self.lens_ms.len() as f64).ceil() as usize).clamp(1, self.lens_ms.len());
                json!(self.lens_ms[k - 1])
            }
        };
        json!({"cleaned_runs": self.lens_ms.len(), "median_ms": q(0.5), "p90_ms": q(0.9),
            "floor_takes": self.takes, "backchannels": self.backchannels,
            "speech_share": self.speech as f64 / self.samples.max(1) as f64})
    }
}

/// One prepared session and Silero's raw runs of its two channels.
pub struct Session {
    pub samples: u64,
    pub redactions: Vec<(u64, u64)>,
    pub dropped: bool,
    pub energy_runs: [Vec<Run>; 2],
    /// `(channel, o, r)` of the policy's pauses.
    pub pauses: Vec<(usize, u64, Option<u64>)>,
    pub silero_runs: [Vec<Run>; 2],
}

/// The measurement over prepared sessions.
pub fn measure_sessions(sessions: &[Session]) -> Value {
    let (mut en, mut si, mut v1, mut v2) = (
        Tally::default(),
        Tally::default(),
        Tally::default(),
        Tally::default(),
    );
    let mut stats: [[RunStats; 2]; 2] = Default::default();
    let mut x = Crosscheck::default();
    let mut dropped = 0;
    for s in sessions {
        if s.dropped {
            dropped += 1;
            continue;
        }
        let n = s.samples;
        let e = [
            Channel::new(&s.energy_runs[0], n),
            Channel::new(&s.energy_runs[1], n),
        ];
        let g = [
            Channel::new(&s.silero_runs[0], n),
            Channel::new(&s.silero_runs[1], n),
        ];
        let w = [
            channel_with_gap(&s.energy_runs[0], n, 300 * MS),
            channel_with_gap(&s.energy_runs[1], n, 300 * MS),
        ];
        for ch in 0..2 {
            stats[0][ch].add(&e[ch]);
            stats[1][ch].add(&g[ch]);
        }
        let mut eligible = Vec::new();
        for &(a, o, r) in &s.pauses {
            let out = label(&e[a], &e[1 - a], o, r, &s.redactions);
            en.add(out);
            if let Ok(l) = out {
                eligible.push((a, o, l.eot));
            }
            v1.add(label(&w[a], &w[1 - a], o, r, &s.redactions));
            v2.add(label(&e[a], &g[1 - a], o, r, &s.redactions));
        }
        for a in 0..2 {
            for (o, r) in silero_offsets(&g[a]) {
                si.add(label(&g[a], &g[1 - a], o, r, &s.redactions));
            }
        }
        crosscheck_session(
            &eligible,
            [s.silero_runs[0].as_slice(), s.silero_runs[1].as_slice()],
            n,
            &s.redactions,
            &mut x,
        );
    }
    let mut runs = serde_json::Map::new();
    for (k, name) in ["energy", "silero"].iter().enumerate() {
        runs.insert(
            name.to_string(),
            json!({"channel_0": stats[k][0].json(), "channel_1": stats[k][1].json()}),
        );
    }
    let xr = x.report();
    json!({
        "sessions": sessions.len(), "sessions_dropped_by_the_session_rules": dropped,
        "energy": en.json(), "silero": si.json(), "runs": runs,
        "crosscheck": {"agreement": xr["agreement"], "table": xr["table"], "unconfirmed": xr["unconfirmed"],
            "eligible_energy_records": xr["eligible_energy_records"], "time_difference_ms": xr["time_difference_ms"]},
        "variant_i_gap_300ms": v1.json(), "variant_ii_b_from_silero": v2.json(),
    })
}

/// `measure`: DIR/sessions.jsonl and DIR/silero-meas.jsonl into DIR/MEASURE.json.
pub fn measure(dir: &Path) -> Result<Value> {
    let mut sil: HashMap<(String, u64), Vec<Run>> = HashMap::new();
    for line in BufReader::new(File::open(dir.join("silero-meas.jsonl"))?).lines() {
        let v: Value = serde_json::from_str(&line?)?;
        sil.insert(
            (
                v["session"].as_str().unwrap_or("").to_string(),
                v["channel"].as_u64().unwrap_or(9),
            ),
            runs_of_json(&v["runs"]),
        );
    }
    let mut sessions = Vec::new();
    for line in BufReader::new(File::open(dir.join("sessions.jsonl"))?).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(&line)?;
        let id = v["session"].as_str().unwrap_or("").to_string();
        let get = |ch: u64| {
            sil.get(&(id.clone(), ch))
                .cloned()
                .ok_or_else(|| anyhow!("no Silero runs of {id}:{ch}"))
        };
        sessions.push(Session {
            samples: v["samples"].as_u64().unwrap_or(0),
            redactions: v["redactions"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|p| (p[0].as_u64().unwrap_or(0), p[1].as_u64().unwrap_or(0)))
                .collect(),
            dropped: !v["drop"].is_null(),
            energy_runs: [
                runs_of_json(&v["energy_runs"][0]),
                runs_of_json(&v["energy_runs"][1]),
            ],
            pauses: v["pauses"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|p| {
                    (
                        p["channel"].as_u64().unwrap_or(0) as usize,
                        p["o"].as_u64().unwrap_or(0),
                        p["r"].as_u64(),
                    )
                })
                .collect(),
            silero_runs: [get(0)?, get(1)?],
        });
    }
    let v = measure_sessions(&sessions);
    fs::write(
        dir.join("MEASURE.json"),
        serde_json::to_string_pretty(&v)? + "\n",
    )?;
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conv::synth_session;

    #[test]
    fn both_paths_agree_when_the_detectors_do() {
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
        let s = Session {
            samples: p.n,
            redactions: Vec::new(),
            dropped: p.drop.is_some(),
            energy_runs: p.raw_runs.clone(),
            pauses: p
                .candidates
                .iter()
                .map(|c| (c.channel, c.pause.offset, c.pause.resume))
                .collect(),
            silero_runs: p.raw_runs.clone(),
        };
        let v = measure_sessions(&[s]);
        assert_eq!(v["crosscheck"]["agreement"]["value"], 1.0, "{v}");
        // the same runs for B: variant (ii) is the energy path
        assert_eq!(v["variant_ii_b_from_silero"]["eot"], v["energy"]["eot"]);
        assert_eq!(v["energy"]["eot"], v["silero"]["eot"]);
        assert_eq!(v["energy"]["candidates"], p.candidates.len());
        // the same runs with a 150 ms fill are Channel::new's
        let c = channel_with_gap(&p.raw_runs[0], p.n, CLEAN_MS * MS);
        let d = Channel::new(&p.raw_runs[0], p.n);
        assert_eq!((c.runs, c.kinds), (d.runs, d.kinds));
        assert!(
            v["runs"]["energy"]["channel_0"]["floor_takes"]
                .as_u64()
                .unwrap()
                > 0
        );
    }
}
