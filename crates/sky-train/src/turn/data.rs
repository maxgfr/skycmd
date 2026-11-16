//! s6-ear v2 data (docs/s6-ear2-prereg.md §3.3, §4.5): a from-disk loader over the feature store
//! written by sky-data (`fetch-turn`: f16 raw hops; `scripts/turn-teacher.py`: u8 teacher states),
//! the student's view of a clip (context jitter, normalisation, masks) and the batches.
//!
//! Sampling gives every training clip the same probability: shuffled epochs, each a function of
//! the seed and the epoch, so the batch of a step is the same after a resume.

use std::path::Path;

use anyhow::{bail, Result};
use mlx_rs::Array;
use sky_infer::audio::{normalize_window, N_MELS, TURN_CHANNELS};
use sky_infer::turn::store::{Clip, Store};
use sky_infer::turn::{TEACHER_DIM, TEACHER_STEPS};

use super::config::TurnConfig;
use super::model::TurnBatch;
use crate::rng::{hash_str, Rng};

/// The light augmentation of §3.3, on the student's view only.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Augment {
    /// Probability that the window's start moves so that 1–4 s are kept.
    pub jitter: f64,
    /// Shortest kept window of the jitter, in hops (1 s).
    pub jitter_min_hops: usize,
    /// Frequency masks per window and their largest width (mel bands).
    pub freq_masks: usize,
    pub freq_width: usize,
    /// Largest width of the one time mask (hops), kept out of the window's last `time_guard` hops.
    pub time_width: usize,
    pub time_guard: usize,
}

impl Augment {
    /// §3.3: jitter 0.3 (1–4 s), two frequency masks of at most 4 bands, one time mask of at
    /// most 8 hops in the first 3 s (never in the last second).
    pub const PREREG: Augment = Augment {
        jitter: 0.3,
        jitter_min_hops: 100,
        freq_masks: 2,
        freq_width: 4,
        time_width: 8,
        time_guard: 100,
    };
}

/// The training and validation clips of a store.
pub struct Data {
    pub store: Store,
    pub train: Vec<usize>,
    pub val: Vec<usize>,
    /// Clips left out: too short for one position, or without a label.
    pub skipped: usize,
}

impl Data {
    /// Opens a store; `max_train` / `max_val` keep the first clips of each split (0: all).
    pub fn open(dir: &Path, cfg: &TurnConfig, max_train: usize, max_val: usize) -> Result<Data> {
        let store = Store::open(dir)?;
        let (mut train, mut val, mut skipped) = (Vec::new(), Vec::new(), 0);
        for (i, c) in store.clips.iter().enumerate() {
            if c.hops < cfg.hops_per_position || c.label().is_none() {
                skipped += 1;
                continue;
            }
            match c.split() {
                "train" => train.push(i),
                "val" => val.push(i),
                _ => skipped += 1,
            }
        }
        if max_train > 0 {
            train.truncate(max_train);
        }
        if max_val > 0 {
            val.truncate(max_val);
        }
        if train.is_empty() {
            bail!("{}: no training clip", dir.display());
        }
        Ok(Data {
            store,
            train,
            val,
            skipped,
        })
    }

    /// Clips of the store whose part has the teacher's targets.
    pub fn with_teacher(&self, idx: &[usize]) -> usize {
        idx.iter()
            .filter(|&&i| self.store.has_teacher(&self.store.clips[i].tag))
            .count()
    }
}

/// The shuffled order of the training clips, epoch by epoch.
pub struct Order {
    seed: u64,
    n: usize,
    epoch: u64,
    perm: Vec<usize>,
}

impl Order {
    pub fn new(seed: u64, n: usize) -> Self {
        let mut o = Order {
            seed,
            n,
            epoch: u64::MAX,
            perm: Vec::new(),
        };
        o.set_epoch(0);
        o
    }

    fn set_epoch(&mut self, e: u64) {
        if e == self.epoch {
            return;
        }
        self.perm = (0..self.n).collect();
        Rng::new(self.seed ^ hash_str(&format!("turn/epoch/{e}"))).shuffle(&mut self.perm);
        self.epoch = e;
    }

    /// The training slots of a step's batch.
    pub fn batch(&mut self, step: u64, batch: usize) -> Vec<usize> {
        (0..batch as u64)
            .map(|r| {
                let k = step * batch as u64 + r;
                self.set_epoch(k / self.n as u64);
                self.perm[(k % self.n as u64) as usize]
            })
            .collect()
    }

    /// Epochs done by `samples` samples.
    pub fn epochs(&self, samples: u64) -> f64 {
        samples as f64 / self.n as f64
    }
}

/// A batch on the host.
#[derive(Debug, Clone, Default)]
pub struct HostBatch {
    pub x: Vec<f32>,
    pub valid: Vec<bool>,
    pub y: Vec<f32>,
    /// The teacher's logit (0 without teacher).
    pub zt: Vec<f32>,
    pub et: Vec<f32>,
    pub st: Vec<f32>,
    pub smask: Vec<f32>,
    pub tmask: Vec<f32>,
    /// Positions of each row.
    pub positions: Vec<usize>,
    pub rows: usize,
}

/// The student's view of one clip: `[positions][input_width]` normalized values and the count.
/// With `aug`, the context jitter, then the masks (normalized value 0) after the normalisation.
pub fn view(
    cfg: &TurnConfig,
    raw: &[f32],
    n: usize,
    aug: Option<(&Augment, &mut Rng)>,
) -> (Vec<f32>, usize) {
    let hpp = cfg.hops_per_position;
    let n = n.min(cfg.window_hops);
    let raw = &raw[(raw.len() / TURN_CHANNELS - n) * TURN_CHANNELS..];
    let Some((a, rng)) = aug else {
        return normalize_window(raw, n, hpp);
    };
    let mut keep = n;
    if rng.f64() < a.jitter {
        let lo = a.jitter_min_hops.div_ceil(hpp);
        let hi = cfg.max_positions();
        let p = lo + rng.below(hi - lo + 1);
        keep = keep.min(p * hpp);
    }
    let (mut x, positions) = normalize_window(&raw[(n - keep) * TURN_CHANNELS..], keep, hpp);
    let hops = positions * hpp;
    for _ in 0..a.freq_masks {
        let w = rng.below(a.freq_width + 1);
        let f = rng.below(N_MELS - w + 1);
        for h in 0..hops {
            x[h * TURN_CHANNELS + f..h * TURN_CHANNELS + f + w].fill(0.0);
        }
    }
    let w = rng.below(a.time_width + 1);
    if w > 0 && hops >= a.time_guard + w {
        let start = rng.below(hops - a.time_guard - w + 1);
        x[start * TURN_CHANNELS..(start + w) * TURN_CHANNELS].fill(0.0);
    }
    (x, positions)
}

/// Builds a batch of clips (`aug` and its generator for training).
pub fn host_batch(
    cfg: &TurnConfig,
    store: &Store,
    clips: &[&Clip],
    mut aug: Option<(&Augment, u64)>,
) -> Result<HostBatch> {
    let (t, wd) = (cfg.max_positions(), cfg.input_width());
    let pps = cfg.positions_per_step();
    let rows = clips.len();
    let mut b = HostBatch {
        x: vec![0.0; rows * t * wd],
        valid: vec![false; rows * t],
        y: vec![0.0; rows],
        zt: vec![0.0; rows],
        et: vec![0.0; rows * TEACHER_DIM],
        st: vec![0.0; rows * TEACHER_STEPS * TEACHER_DIM],
        smask: vec![0.0; rows * TEACHER_STEPS],
        tmask: vec![0.0; rows],
        positions: vec![0; rows],
        rows,
    };
    for (r, c) in clips.iter().enumerate() {
        let raw = store.hops(c)?;
        let (x, p) = match aug.as_mut() {
            Some((a, seed)) => {
                let mut rng = Rng::new(*seed ^ hash_str(&format!("turn/aug/{r}")));
                view(cfg, &raw, c.hops, Some((a, &mut rng)))
            }
            None => view(cfg, &raw, c.hops, None),
        };
        b.x[(r * t + t - p) * wd..(r + 1) * t * wd].copy_from_slice(&x);
        b.valid[r * t + t - p..(r + 1) * t].fill(true);
        b.positions[r] = p;
        b.y[r] = c.label().unwrap_or(false) as u8 as f32;
        if let Some(tc) = store.teacher(c)? {
            b.tmask[r] = 1.0;
            b.zt[r] = tc.logit;
            b.et[r * TEACHER_DIM..(r + 1) * TEACHER_DIM].copy_from_slice(&tc.pooled);
            b.st[r * TEACHER_STEPS * TEACHER_DIM..(r + 1) * TEACHER_STEPS * TEACHER_DIM]
                .copy_from_slice(&tc.states);
            for s in 0..TEACHER_STEPS {
                b.smask[r * TEACHER_STEPS + s] = (s * pps >= t - p) as u8 as f32;
            }
        }
    }
    Ok(b)
}

impl HostBatch {
    pub fn arrays(&self, cfg: &TurnConfig) -> TurnBatch {
        let (b, t, wd) = (
            self.rows as i32,
            cfg.max_positions() as i32,
            cfg.input_width() as i32,
        );
        let (td, ts) = (TEACHER_DIM as i32, TEACHER_STEPS as i32);
        let q: Vec<f32> = self
            .zt
            .iter()
            .map(|&z| sky_infer::turn::model::sigmoid(z))
            .collect();
        TurnBatch {
            x: Array::from_slice(&self.x, &[b, t, wd]),
            valid: Array::from_slice(&self.valid, &[b, t]),
            y: Array::from_slice(&self.y, &[b]),
            q: Array::from_slice(&q, &[b]),
            et: Array::from_slice(&self.et, &[b, td]),
            st: Array::from_slice(&self.st, &[b, ts, td]),
            smask: Array::from_slice(&self.smask, &[b, ts]),
            tmask: Array::from_slice(&self.tmask, &[b]),
            rows: self.rows,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sky_infer::audio::{TurnFrontEnd, CH_ENERGY, CH_F0, CH_VOICING};

    fn femto() -> TurnConfig {
        TurnConfig::load(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/s6-ear2-femto.json"),
            None,
        )
        .unwrap()
    }

    fn hops(secs: f32) -> (Vec<f32>, usize) {
        let pcm: Vec<f32> = (0..(16_000.0 * secs) as usize)
            .map(|i| 0.3 * (i as f32 * 0.07).sin() + 0.1 * (i as f32 * 0.013).sin())
            .collect();
        TurnFrontEnd::new().hops(&pcm, 400)
    }

    #[test]
    fn the_view_jitters_and_masks_away_from_the_end() {
        let cfg = femto();
        let (raw, n) = hops(5.0);
        assert_eq!(n, 400);
        let (clean, p) = view(&cfg, &raw, n, None);
        assert_eq!(p, 100);
        let a = Augment::PREREG;
        let (mut jittered, mut lens) = (0, Vec::new());
        for s in 0..400u64 {
            let mut rng = Rng::new(s);
            let (x, p) = view(&cfg, &raw, n, Some((&a, &mut rng)));
            assert_eq!(x.len(), p * cfg.input_width());
            assert!((25..=100).contains(&p));
            if p < 100 {
                jittered += 1;
            }
            lens.push(p);
            // the window always ends at the clip's end, and the last second is never time-masked
            let last = &x[(p * 4 - 100) * TURN_CHANNELS..];
            let clean_last = &clean[(400 - 100) * TURN_CHANNELS..];
            if p == 100 {
                for (h, (u, v)) in last
                    .as_chunks::<TURN_CHANNELS>()
                    .0
                    .iter()
                    .zip(clean_last.as_chunks::<TURN_CHANNELS>().0.iter())
                    .enumerate()
                {
                    // only frequency masks there: f0, voicing and energy untouched
                    assert_eq!(
                        (u[CH_F0], u[CH_VOICING], u[CH_ENERGY]),
                        (v[CH_F0], v[CH_VOICING], v[CH_ENERGY]),
                        "hop {h}"
                    );
                    let masked = u[..N_MELS]
                        .iter()
                        .zip(&v[..N_MELS])
                        .filter(|(a, b)| a != b)
                        .count();
                    assert!(masked <= 8);
                }
            }
        }
        // about 30% jittered (a draw of 100 positions keeps the whole window)
        assert!((90..150).contains(&jittered), "{jittered}");
        assert!(lens.iter().any(|&p| p < 40));
    }

    #[test]
    fn epochs_visit_every_clip_once() {
        let mut o = Order::new(3, 10);
        let mut seen: Vec<usize> = (0..5).flat_map(|s| o.batch(s, 4)).collect();
        assert_eq!(seen.len(), 20);
        let first: Vec<usize> = seen.drain(..10).collect();
        let mut sorted = first.clone();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
        // the same step gives the same batch in a new order (a resume)
        assert_eq!(Order::new(3, 10).batch(3, 4), o.batch(3, 4));
        assert!((o.epochs(25) - 2.5).abs() < 1e-12);
    }
}
