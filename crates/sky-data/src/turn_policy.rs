//! s6-ear v3 (docs/s6-ear3-prereg.md §3.2 item 2): a Rust port of the end-of-turn demo's energy
//! voice-activity detector and turn policy (`EnergyVad` and `TurnPolicy` of
//! `site/.vitepress/theme/lib/turn.ts`, with `TURN_DEFAULTS`), checked bit for bit against vectors
//! recorded from the TypeScript by `scripts/turn-policy-vectors.mjs`
//! (`crates/sky-data/tests/fixtures/turn-policy.json`).
//!
//! The arithmetic follows JavaScript's: samples are `f32` read as `f64`, the RMS, the noise floor
//! and the threshold are `f64`, the clock is whole milliseconds. [`replay`] runs the pair over a
//! 16 kHz channel with a model that never answers "finished", so that every ask the policy would
//! make inside a pause is produced; [`pauses`] turns its events into the pauses of §3.2: the
//! offset, the asks and the next onset.

/// One 10 ms frame at 16 kHz.
pub const FRAME_SAMPLES: usize = 160;
/// Samples per millisecond at 16 kHz.
pub const SAMPLES_PER_MS: u64 = 16;

/// `TURN_DEFAULTS` of `turn.ts`.
pub const MIN_SPEECH_MS: u64 = 150;
pub const STOP_MS: u64 = 200;
pub const RECHECK_MS: u64 = 200;
pub const FALLBACK_MS: u64 = 2500;
pub const FRAME_MS: u64 = 10;
pub const TAU: f64 = 0.5;
pub const ABSOLUTE_FLOOR: f64 = 0.008;
pub const RATIO: f64 = 3.5;
/// `EnergyVad`'s defaults: the floor follows quiet frames at this rate, from this value, and never
/// goes below `FLOOR_MIN`.
pub const ADAPT: f64 = 0.05;
pub const INITIAL_FLOOR: f64 = 0.002;
pub const FLOOR_MIN: f64 = 1e-4;

/// `EnergyVad`: the RMS of each frame against a noise floor tracked slowly on quiet frames.
#[derive(Clone, Debug)]
pub struct EnergyVad {
    /// RMS of the last frame.
    pub rms: f64,
    pub noise_floor: f64,
    absolute_floor: f64,
    ratio: f64,
    adapt: f64,
}

impl Default for EnergyVad {
    fn default() -> Self {
        Self::new()
    }
}

impl EnergyVad {
    pub fn new() -> Self {
        EnergyVad {
            rms: 0.0,
            noise_floor: INITIAL_FLOOR,
            absolute_floor: ABSOLUTE_FLOOR,
            ratio: RATIO,
            adapt: ADAPT,
        }
    }

    /// Whether one frame is speech; a quiet frame moves the floor (`isSpeech`).
    pub fn is_speech(&mut self, frame: &[f32]) -> bool {
        let mut sum = 0.0f64;
        for &x in frame {
            let x = x as f64;
            sum += x * x;
        }
        self.rms = (sum / frame.len().max(1) as f64).sqrt();
        let speech = self.rms > self.absolute_floor.max(self.ratio * self.noise_floor);
        if !speech {
            self.noise_floor += self.adapt * (self.rms - self.noise_floor);
        }
        self.noise_floor = FLOOR_MIN.max(self.noise_floor);
        speech
    }
}

/// `TurnState` (the `finished` state is never kept).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Waiting,
    Speaking,
    Holding,
}

/// What a `TurnEvent` is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    SpeechStart,
    /// The model was asked; its probability.
    Decision(f64),
    /// The turn ended: by the model (with its probability) or by the timer.
    Finished {
        by_model: bool,
    },
    Resumed,
}

/// One event, at a millisecond of the audio timeline (the end of the frame that caused it).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Event {
    pub kind: Kind,
    pub at_ms: u64,
}

/// `TurnPolicy` with `TURN_DEFAULTS`, fed one frame at a time.
#[derive(Clone, Debug)]
pub struct TurnPolicy {
    pub state: State,
    pub t_ms: u64,
    pub tau: f64,
    speech_ms: u64,
    silence_ms: u64,
    since_ask_ms: u64,
}

impl Default for TurnPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl TurnPolicy {
    pub fn new() -> Self {
        TurnPolicy {
            state: State::Waiting,
            t_ms: 0,
            tau: TAU,
            speech_ms: 0,
            silence_ms: 0,
            since_ask_ms: 0,
        }
    }

    /// One frame (`feed`): `speech` is the VAD's answer; `prob_end` is called only when a
    /// decision is due. The events are appended to `out` in order.
    pub fn feed(&mut self, speech: bool, prob_end: &mut dyn FnMut() -> f64, out: &mut Vec<Event>) {
        self.t_ms += FRAME_MS;
        if speech {
            self.silence_ms = 0;
            self.since_ask_ms = 0;
            match self.state {
                State::Waiting => {
                    self.speech_ms += FRAME_MS;
                    if self.speech_ms >= MIN_SPEECH_MS {
                        self.state = State::Speaking;
                        out.push(Event {
                            kind: Kind::SpeechStart,
                            at_ms: self.t_ms,
                        });
                    }
                }
                State::Holding => {
                    self.state = State::Speaking;
                    out.push(Event {
                        kind: Kind::Resumed,
                        at_ms: self.t_ms,
                    });
                }
                State::Speaking => {}
            }
            return;
        }
        self.speech_ms = 0;
        if self.state == State::Waiting {
            return;
        }
        self.silence_ms += FRAME_MS;
        self.since_ask_ms += FRAME_MS;
        if self.silence_ms >= FALLBACK_MS {
            self.finish(false, out);
            return;
        }
        let due = if self.state == State::Speaking {
            self.silence_ms >= STOP_MS
        } else {
            self.since_ask_ms >= RECHECK_MS
        };
        if due {
            let p = prob_end();
            self.since_ask_ms = 0;
            out.push(Event {
                kind: Kind::Decision(p),
                at_ms: self.t_ms,
            });
            if p >= self.tau {
                self.finish(true, out);
            } else {
                self.state = State::Holding;
            }
        }
    }

    fn finish(&mut self, by_model: bool, out: &mut Vec<Event>) {
        self.state = State::Waiting;
        self.speech_ms = 0;
        self.silence_ms = 0;
        self.since_ask_ms = 0;
        out.push(Event {
            kind: Kind::Finished { by_model },
            at_ms: self.t_ms,
        });
    }
}

/// The replay of one channel: the VAD's answer per frame and the policy's events.
#[derive(Clone, Debug, Default)]
pub struct Replay {
    pub speech: Vec<bool>,
    /// RMS of each frame (the bleed filter reads it).
    pub rms: Vec<f64>,
    pub events: Vec<Event>,
}

/// Runs the VAD and the policy over every whole 10 ms frame of 16 kHz samples, with a model that
/// never answers "finished" (probability 0).
pub fn replay(pcm: &[f32]) -> Replay {
    let mut vad = EnergyVad::new();
    let mut policy = TurnPolicy::new();
    let n = pcm.len() / FRAME_SAMPLES;
    let mut r = Replay {
        speech: Vec::with_capacity(n),
        rms: Vec::with_capacity(n),
        events: Vec::new(),
    };
    let mut never = || 0.0;
    for f in pcm.as_chunks::<FRAME_SAMPLES>().0 {
        let s = vad.is_speech(f);
        r.speech.push(s);
        r.rms.push(vad.rms);
        policy.feed(s, &mut never, &mut r.events);
    }
    r
}

/// One pause of the policy with at least one ask (§3.2 item 2), in 16 kHz samples.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pause {
    /// The end of the last speech frame before the pause (the first ask is at `offset + stopMs`).
    pub offset: u64,
    /// The asks, in ms of silence after the offset: 200, 400, … (at most 2,400).
    pub asks: Vec<u64>,
    /// The next onset of the speaker: the start of the frame of the policy's `resumed`, or, after
    /// the fallback timer ended the pause, the start of the run of speech frames that gave the
    /// next `speech-start`; `None` when the channel ends first.
    pub resume: Option<u64>,
}

/// The pauses of a replay (docs/s6-ear3-prereg.md §3.2 item 2 and amendment 1).
pub fn pauses(events: &[Event]) -> Vec<Pause> {
    let mut out: Vec<Pause> = Vec::new();
    let mut open: Option<Pause> = None;
    // a pause ended by the timer waits for the next speech-start for its onset
    let mut timed_out: Option<Pause> = None;
    for e in events {
        match e.kind {
            Kind::Decision(_) => {
                let p = open.get_or_insert_with(|| Pause {
                    offset: (e.at_ms - STOP_MS) * SAMPLES_PER_MS,
                    asks: Vec::new(),
                    resume: None,
                });
                p.asks.push(e.at_ms - p.offset / SAMPLES_PER_MS);
            }
            Kind::Resumed => {
                if let Some(mut p) = open.take() {
                    p.resume = Some((e.at_ms - FRAME_MS) * SAMPLES_PER_MS);
                    out.push(p);
                }
            }
            Kind::Finished { by_model } => {
                if let Some(p) = open.take() {
                    if by_model {
                        out.push(p);
                    } else {
                        timed_out = Some(p);
                    }
                }
            }
            Kind::SpeechStart => {
                if let Some(mut p) = timed_out.take() {
                    p.resume = Some((e.at_ms - MIN_SPEECH_MS) * SAMPLES_PER_MS);
                    out.push(p);
                }
            }
        }
    }
    out.extend(open);
    out.extend(timed_out);
    out.sort_by_key(|p| p.offset);
    out
}

/// The samples of a recorded vector (`scripts/turn-policy-vectors.mjs`): its segments
/// `[frames, amplitude]`, each sample the amplitude times a 32-bit LCG value in [-1, 1), rounded
/// to f32 as the Float32Array rounds it.
fn vector_frames(v: &serde_json::Value) -> Option<Vec<Vec<f32>>> {
    let mut s = v["seed"].as_u64()? as u32;
    let mut noise = move || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        (s >> 8) as f64 / 16_777_216.0 * 2.0 - 1.0
    };
    let mut out = Vec::new();
    for seg in v["segments"].as_array()? {
        let (n, amp) = (seg[0].as_u64()?, seg[1].as_f64()?);
        for _ in 0..n {
            out.push((0..FRAME_SAMPLES).map(|_| (amp * noise()) as f32).collect());
        }
    }
    Some(out)
}

/// The check of §8.1 item 1: replays the vectors recorded from `turn.ts` (the fixture
/// `crates/sky-data/tests/fixtures/turn-policy.json`) through the port; `passed` when every
/// frame's answer, RMS and noise floor and every event (its kind, frame and probability) are the
/// same, bit for bit. The first mismatch of each vector is reported.
pub fn check_vectors(path: &std::path::Path) -> anyhow::Result<serde_json::Value> {
    use anyhow::Context;
    let fx: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(path).with_context(|| path.display().to_string())?,
    )?;
    if fx["frameSamples"].as_u64() != Some(FRAME_SAMPLES as u64) {
        anyhow::bail!(
            "{}: frames of {} samples",
            path.display(),
            fx["frameSamples"]
        );
    }
    let mut report = Vec::new();
    let mut passed = true;
    for v in fx["vectors"].as_array().context("no vectors")? {
        let name = v["name"].as_str().unwrap_or("");
        let probs: Vec<f64> = v["probs"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|p| p.as_f64())
            .collect();
        let mut ask = 0usize;
        let mut prob = || {
            let p = probs.get(ask).copied().unwrap_or(0.0);
            ask += 1;
            p
        };
        let (mut vad, mut policy, mut events) = (EnergyVad::new(), TurnPolicy::new(), Vec::new());
        let want_speech = v["speech"].as_str().unwrap_or("").as_bytes();
        let fr = vector_frames(v).context("a vector without segments")?;
        let mut first: Option<String> = None;
        if fr.len() != want_speech.len() {
            first = Some(format!(
                "{} frames, {} recorded",
                fr.len(),
                want_speech.len()
            ));
        }
        for (k, f) in fr.iter().enumerate() {
            let s = vad.is_speech(f);
            if first.is_none()
                && (want_speech.get(k).map(|&c| c == b'1') != Some(s)
                    || v["rms"][k].as_f64() != Some(vad.rms)
                    || v["noiseFloor"][k].as_f64() != Some(vad.noise_floor))
            {
                first = Some(format!("frame {k}"));
            }
            policy.feed(s, &mut prob, &mut events);
        }
        let want = v["events"].as_array().cloned().unwrap_or_default();
        if first.is_none() && events.len() != want.len() {
            first = Some(format!("{} events, {} recorded", events.len(), want.len()));
        }
        for (i, (e, w)) in events.iter().zip(&want).enumerate() {
            let ty = w["type"].as_str().unwrap_or("");
            let same = w["atMs"].as_u64() == Some(e.at_ms)
                && match e.kind {
                    Kind::SpeechStart => ty == "speech-start",
                    Kind::Resumed => ty == "resumed",
                    Kind::Decision(p) => ty == "decision" && w["p"].as_f64() == Some(p),
                    Kind::Finished { by_model } => {
                        ty == "finished" && (w["by"] == "model") == by_model
                    }
                };
            if !same && first.is_none() {
                first = Some(format!("event {i}"));
            }
        }
        passed &= first.is_none();
        report.push(
            serde_json::json!({"vector": name, "frames": fr.len(), "events": events.len(),
            "first_mismatch": first}),
        );
    }
    Ok(
        serde_json::json!({"fixture": path, "passed": passed, "vectors": report,
        "rule": "docs/s6-ear3-prereg.md §8.1 item 1: the port replays the recorded turn.ts vectors bit for bit"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_port_replays_turn_ts_bit_for_bit() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/turn-policy.json");
        let r = check_vectors(&p).unwrap();
        assert_eq!(r["passed"], true, "{r}");
        assert_eq!(r["vectors"].as_array().unwrap().len(), 3);
        // a changed recording is caught
        let mut fx: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
        fx["vectors"][0]["events"][0]["atMs"] = serde_json::json!(1);
        let t = std::env::temp_dir().join(format!("turn-policy-{}.json", std::process::id()));
        std::fs::write(&t, fx.to_string()).unwrap();
        let r = check_vectors(&t).unwrap();
        assert_eq!(r["passed"], false);
        assert_eq!(r["vectors"][0]["first_mismatch"], "event 0");
        std::fs::remove_file(t).ok();
    }

    #[test]
    fn pauses_give_offsets_asks_and_onsets() {
        // speech, a pause of 450 ms (two asks), speech, a pause past the timer, a blip, speech
        let mut sp = vec![true; 30];
        sp.extend(vec![false; 45]);
        sp.extend(vec![true; 20]);
        sp.extend(vec![false; 300]);
        sp.extend(vec![true; 5]); // under minSpeechMs: no speech-start
        sp.extend(vec![false; 3]);
        sp.extend(vec![true; 20]);
        sp.extend(vec![false; 30]);
        let (mut policy, mut ev) = (TurnPolicy::new(), Vec::new());
        let mut never = || 0.0;
        for &s in &sp {
            policy.feed(s, &mut never, &mut ev);
        }
        let p = pauses(&ev);
        assert_eq!(p.len(), 3, "{p:?}");
        // the first pause: offset at frame 30's end (300 ms), asks at 200 and 400, resumed at 750
        assert_eq!(p[0].offset, 300 * 16);
        assert_eq!(p[0].asks, vec![200, 400]);
        assert_eq!(p[0].resume, Some(750 * 16));
        // the second: 12 asks up to 2,400 ms, the timer at 2,500, then the onset of the 20-frame
        // run (the 5-frame blip is not a speech-start)
        assert_eq!(p[1].offset, 950 * 16);
        assert_eq!(p[1].asks, (1..=12).map(|k| 200 * k).collect::<Vec<_>>());
        assert_eq!(p[1].resume, Some((950 + 3000 + 50 + 30) * 16));
        // the last: the channel ends during the pause
        assert_eq!(p[2].asks, vec![200]);
        assert_eq!(p[2].resume, None);
    }

    #[test]
    fn replay_reads_whole_frames() {
        let mut pcm = vec![0.0f32; 160 * 60];
        for x in pcm[160 * 5..160 * 30].iter_mut() {
            *x = 0.3;
        }
        pcm.extend([0.5f32; 100]); // a partial frame is not read
        let r = replay(&pcm);
        assert_eq!(r.speech.len(), 60);
        assert!(r.speech[5..30].iter().all(|&s| s) && !r.speech[30]);
        assert_eq!(pauses(&r.events)[0].offset, 300 * 16);
    }
}
