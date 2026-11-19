//! s6-ear v2 data (docs/s6-ear2-prereg.md §3.3, §4.5): a from-disk loader over the feature store
//! written by sky-data (`fetch-turn`: f16 raw hops; `scripts/turn-teacher.py`: u8 teacher states),
//! the student's view of a clip (context jitter, normalisation, masks) and the batches.
//!
//! Sampling gives every training clip the same probability: shuffled epochs, each a function of
//! the seed and the epoch, so the batch of a step is the same after a resume.
//!
//! s6-ear v3 (docs/s6-ear3-prereg.md §3.4, §3.6, §16) adds the second source and the trailing
//! silence: [`Conv`] (the conversation records of `sky-data fetch-conv`, read at a cut o + d),
//! [`Ext`] (the 240 hops after each Smart Turn clip's end, `sky-data turn-extend`), the asks and
//! their weights w(d), and [`mixed_batch`]: each row a conversation record with probability
//! `conv_share` (its class drawn first, 0.5 each, then a record of it, then d from w restricted to
//! its valid asks), else a Smart Turn clip of the epoch order (d = 0 with probability `st_d0`, else
//! d from w). Every draw is a function of the seed, the step and the row.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Result};
use mlx_rs::Array;
use sky_infer::audio::{normalize_window, N_MELS, TURN_CHANNELS};
use sky_infer::turn::store::{Clip, Store, Teacher};
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
            .map(|r| self.slot(step * batch as u64 + r))
            .collect()
    }

    /// The training slot of the `k`-th draw.
    pub fn slot(&mut self, k: u64) -> usize {
        self.set_epoch(k / self.n as u64);
        self.perm[(k % self.n as u64) as usize]
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
    /// s6-ear v3: the weight of each row's label term (see `TurnBatch::hw`).
    pub hw: Option<Vec<f32>>,
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

impl HostBatch {
    /// An empty batch of `rows` rows.
    pub fn new(cfg: &TurnConfig, rows: usize) -> HostBatch {
        let (t, wd) = (cfg.max_positions(), cfg.input_width());
        HostBatch {
            x: vec![0.0; rows * t * wd],
            valid: vec![false; rows * t],
            y: vec![0.0; rows],
            zt: vec![0.0; rows],
            et: vec![0.0; rows * TEACHER_DIM],
            st: vec![0.0; rows * TEACHER_STEPS * TEACHER_DIM],
            smask: vec![0.0; rows * TEACHER_STEPS],
            tmask: vec![0.0; rows],
            positions: vec![0; rows],
            hw: None,
            rows,
        }
    }

    /// Row `r`: the view of the raw hops `raw` (`[n][43]`, the window's last 4 s taken from
    /// their end), its label, and the teacher's targets when given (the state term only when
    /// `states`).
    #[allow(clippy::too_many_arguments)]
    fn put(
        &mut self,
        cfg: &TurnConfig,
        r: usize,
        raw: &[f32],
        label: bool,
        teacher: Option<Teacher>,
        states: bool,
        aug: Option<(&Augment, &mut Rng)>,
    ) {
        let (t, wd) = (cfg.max_positions(), cfg.input_width());
        let pps = cfg.positions_per_step();
        let (x, p) = view(cfg, raw, raw.len() / TURN_CHANNELS, aug);
        self.x[(r * t + t - p) * wd..(r + 1) * t * wd].copy_from_slice(&x);
        self.valid[r * t + t - p..(r + 1) * t].fill(true);
        self.positions[r] = p;
        self.y[r] = label as u8 as f32;
        if let Some(tc) = teacher {
            self.tmask[r] = 1.0;
            self.zt[r] = tc.logit;
            self.et[r * TEACHER_DIM..(r + 1) * TEACHER_DIM].copy_from_slice(&tc.pooled);
            self.st[r * TEACHER_STEPS * TEACHER_DIM..(r + 1) * TEACHER_STEPS * TEACHER_DIM]
                .copy_from_slice(&tc.states);
            if states {
                for s in 0..TEACHER_STEPS {
                    self.smask[r * TEACHER_STEPS + s] = (s * pps >= t - p) as u8 as f32;
                }
            }
        }
    }
}

/// Builds a batch of clips (`aug` and its generator for training).
pub fn host_batch(
    cfg: &TurnConfig,
    store: &Store,
    clips: &[&Clip],
    mut aug: Option<(&Augment, u64)>,
) -> Result<HostBatch> {
    let mut b = HostBatch::new(cfg, clips.len());
    for (r, c) in clips.iter().enumerate() {
        let raw = store.hops(c)?;
        let teacher = store.teacher(c)?;
        let label = c.label().unwrap_or(false);
        match aug.as_mut() {
            Some((a, seed)) => {
                let mut rng = Rng::new(*seed ^ hash_str(&format!("turn/aug/{r}")));
                b.put(cfg, r, &raw, label, teacher, true, Some((a, &mut rng)));
            }
            None => b.put(cfg, r, &raw, label, teacher, true, None),
        }
    }
    Ok(b)
}

// ----------------------------------------------------------------------------- s6-ear v3

/// The asks of the deployed policy (ms of trailing silence): 0.2, 0.4, …, 2.4 s.
pub const ASKS_MS: [usize; 12] = [
    200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400,
];

/// w(d) of §3.6: 0.5 at the first ask, 0.5 / 11 at each of the eleven others.
pub fn ask_weight(d: usize) -> f64 {
    if d == ASKS_MS[0] {
        0.5
    } else {
        0.5 / 11.0
    }
}

/// d drawn from w restricted to the asks up to `dmax` ms and renormalised.
pub fn draw_ask(rng: &mut Rng, dmax: usize) -> usize {
    let w: Vec<f64> = ASKS_MS
        .iter()
        .map(|&d| if d <= dmax { ask_weight(d) } else { 0.0 })
        .collect();
    if w.iter().all(|&x| x == 0.0) {
        return ASKS_MS[0];
    }
    ASKS_MS[rng.weighted(&w)]
}

/// The hops of a cut record (a conversation record or an eot-bench span) up to its cut at `d`
/// ms: its first `cut_hops + d / 10` stored hops; `None` when d is above its `dmax_ms`.
pub fn cut_view<'a>(c: &Clip, raw: &'a [f32], d: usize) -> Option<&'a [f32]> {
    let cut = c.meta["cut_hops"].as_u64()? as usize;
    let dmax = c.meta["dmax_ms"].as_u64()? as usize;
    if d > dmax {
        return None;
    }
    let end = (cut + d / 10).min(c.hops);
    Some(&raw[..end * TURN_CHANNELS])
}

/// The conversation records (`sky-data fetch-conv`): training records by class, validation
/// records; the test records are never read.
pub struct Conv {
    pub store: Store,
    /// Training records: `[hold, end of turn]`.
    pub train: [Vec<usize>; 2],
    pub val: Vec<usize>,
    pub skipped: usize,
}

impl Conv {
    /// `max_train` / `max_val` keep the first records of each split (0: all; the smoke run).
    pub fn open(dir: &Path, max_train: usize, max_val: usize) -> Result<Conv> {
        let store = Store::open(dir)?;
        let (mut train, mut val, mut skipped) = (Vec::new(), Vec::new(), 0);
        for (i, c) in store.clips.iter().enumerate() {
            let ok = c.label().is_some()
                && c.meta["cut_hops"].is_u64()
                && c.meta["dmax_ms"]
                    .as_u64()
                    .is_some_and(|d| d >= ASKS_MS[0] as u64);
            match (ok, c.split()) {
                (true, "train") => train.push(i),
                (true, "val") => val.push(i),
                _ => skipped += 1,
            }
        }
        if max_train > 0 {
            train.truncate(max_train);
        }
        if max_val > 0 {
            val.truncate(max_val);
        }
        let mut by = [Vec::new(), Vec::new()];
        for i in train {
            by[store.clips[i].label().unwrap_or(false) as usize].push(i);
        }
        if by[0].is_empty() && by[1].is_empty() {
            bail!("{}: no conversation training record", dir.display());
        }
        Ok(Conv {
            store,
            train: by,
            val,
            skipped,
        })
    }
}

/// The Smart Turn extension (`sky-data turn-extend`): the store and its records by clip id.
pub struct Ext {
    pub store: Store,
    pub by_id: HashMap<String, usize>,
}

impl Ext {
    pub fn open(dir: &Path) -> Result<Ext> {
        let store = Store::open(dir)?;
        let by_id = store
            .clips
            .iter()
            .enumerate()
            .map(|(i, c)| (c.id.clone(), i))
            .collect();
        Ok(Ext { store, by_id })
    }

    /// A clip's own hops followed by the first `d / 10` hops of its extension.
    pub fn extended(&self, clip: &Clip, raw: Vec<f32>, d: usize) -> Result<Vec<f32>> {
        if d == 0 {
            return Ok(raw);
        }
        let Some(&i) = self.by_id.get(&clip.id) else {
            bail!(
                "{}: no extension of clip {}",
                self.store.dir.display(),
                clip.id
            );
        };
        let x = self.store.hops(&self.store.clips[i])?;
        let mut out = raw;
        out.extend_from_slice(&x[..(d / 10) * TURN_CHANNELS]);
        Ok(out)
    }
}

/// The mix of §3.6.
#[derive(Debug, Clone, Copy)]
pub struct Mix {
    /// Probability that a row is a conversation record.
    pub conv_share: f64,
    /// Probability that a Smart Turn row is v2's view (d = 0).
    pub st_d0: f64,
}

/// Whether row `r` of step `step` is a conversation record.
pub fn row_is_conv(seed: u64, step: u64, r: usize, share: f64) -> bool {
    Rng::new(seed ^ hash_str(&format!("turn3/src/{step}/{r}"))).f64() < share
}

/// The Smart Turn draws of the steps before `step` (where the epoch order resumes).
pub fn st_draws_before(seed: u64, step: u64, batch: usize, share: f64) -> u64 {
    (0..step)
        .map(|s| {
            (0..batch)
                .filter(|&r| !row_is_conv(seed, s, r, share))
                .count() as u64
        })
        .sum()
}

/// The two sources of a v3 run.
pub struct Sources<'a> {
    pub st: &'a Data,
    pub ext: &'a Ext,
    pub conv: &'a Conv,
    pub mix: Mix,
    /// The weight of a Smart Turn row's label term (the loss weights' `hard`).
    pub st_hard: f32,
}

/// What a v3 row was (tests and logs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Row {
    St { clip: usize, d: usize },
    Conv { record: usize, d: usize },
}

/// The batch of a v3 training step: the rows of [`row_is_conv`], the Smart Turn rows taking the
/// epoch order from its `st_k`-th draw on (advanced here).
#[allow(clippy::too_many_arguments)]
pub fn mixed_batch(
    cfg: &TurnConfig,
    src: &Sources,
    seed: u64,
    step: u64,
    batch: usize,
    aug: Option<&Augment>,
    order: &mut Order,
    st_k: &mut u64,
) -> Result<(HostBatch, Vec<Row>)> {
    let mut b = HostBatch::new(cfg, batch);
    let mut hw = Vec::with_capacity(batch);
    let mut rows = Vec::with_capacity(batch);
    let aseed = seed ^ hash_str(&format!("turn/step/{step}"));
    for r in 0..batch {
        let mut rng = Rng::new(seed ^ hash_str(&format!("turn3/row/{step}/{r}")));
        let mut arng = Rng::new(aseed ^ hash_str(&format!("turn/aug/{r}")));
        let a = aug.map(|a| (a, &mut arng));
        if row_is_conv(seed, step, r, src.mix.conv_share) {
            let cv = src.conv;
            let mut class = (rng.f64() < 0.5) as usize;
            if cv.train[class].is_empty() {
                class = 1 - class;
            }
            let idx = cv.train[class][rng.below(cv.train[class].len())];
            let c = &cv.store.clips[idx];
            let dmax = c.meta["dmax_ms"].as_u64().unwrap_or(0) as usize;
            let d = draw_ask(&mut rng, dmax);
            let raw = cv.store.hops(c)?;
            let v = cut_view(c, &raw, d).expect("d is a valid ask");
            b.put(cfg, r, v, c.label().unwrap_or(false), None, false, a);
            hw.push(1.0);
            rows.push(Row::Conv { record: idx, d });
        } else {
            let st = src.st;
            let slot = order.slot(*st_k);
            *st_k += 1;
            let ci = st.train[slot];
            let c = &st.store.clips[ci];
            let d = if rng.f64() < src.mix.st_d0 {
                0
            } else {
                draw_ask(&mut rng, ASKS_MS[ASKS_MS.len() - 1])
            };
            let raw = src.ext.extended(c, st.store.hops(c)?, d)?;
            let teacher = st.store.teacher(c)?;
            b.put(cfg, r, &raw, c.label().unwrap_or(false), teacher, d == 0, a);
            hw.push(src.st_hard);
            rows.push(Row::St { clip: ci, d });
        }
    }
    b.hw = Some(hw);
    Ok((b, rows))
}

/// A batch of conversation records cut at `d` ms (validation: no augmentation).
pub fn conv_batch(cfg: &TurnConfig, conv: &Conv, idx: &[usize], d: usize) -> Result<HostBatch> {
    let mut b = HostBatch::new(cfg, idx.len());
    for (r, &i) in idx.iter().enumerate() {
        let c = &conv.store.clips[i];
        let raw = conv.store.hops(c)?;
        let Some(v) = cut_view(c, &raw, d) else {
            bail!("record {} has no ask at {d} ms", c.id);
        };
        b.put(cfg, r, v, c.label().unwrap_or(false), None, false, None);
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
        let hw = self.hw.as_ref().map(|h| Array::from_slice(h, &[b]));
        TurnBatch {
            hw,
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

    #[test]
    fn asks_follow_w_restricted_to_the_valid_ones() {
        let mut rng = Rng::new(1);
        let mut n = [0usize; 3];
        for _ in 0..20_000 {
            let d = draw_ask(&mut rng, 600);
            assert!([200, 400, 600].contains(&d));
            n[d / 200 - 1] += 1;
        }
        // w restricted to {0.2, 0.4, 0.6} and renormalised: 0.5 / (0.5 + 2 x 0.5 / 11)
        let want = 0.5 / (0.5 + 1.0 / 11.0);
        assert!((n[0] as f64 / 20_000.0 - want).abs() < 0.015, "{n:?}");
        assert!((ASKS_MS.iter().map(|&d| ask_weight(d)).sum::<f64>() - 1.0).abs() < 1e-12);
    }

    /// s6-ear v3: the views end at their cut, the rows carry their masks, and the mix and the
    /// class balance are those of §3.6; a step's batch is the same after a resume.
    #[test]
    fn v3_rows_end_at_their_cut_with_their_masks() {
        use crate::turn::train::tests::{tiny_conv, tiny_ext, tiny_store};
        let cfg = femto();
        let dir = std::env::temp_dir().join(format!("turn3-data-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        tiny_store(&dir.join("store"), 40);
        tiny_ext(&dir.join("ext"), 40);
        tiny_conv(&dir.join("conv"), 80);
        let st = Data::open(&dir.join("store"), &cfg, 0, 0).unwrap();
        let ext = Ext::open(&dir.join("ext")).unwrap();
        let conv = Conv::open(&dir.join("conv"), 0, 0).unwrap();
        assert!(conv.train[0].len() > 2 * conv.train[1].len());
        // a record's view at d ends d / 10 hops after its cut; above its dmax it has none
        let c = &conv.store.clips[conv.train[0][0]];
        let raw = conv.store.hops(c).unwrap();
        let cut = c.meta["cut_hops"].as_u64().unwrap() as usize;
        assert_eq!(
            cut_view(c, &raw, 600).unwrap().len(),
            (cut + 60) * TURN_CHANNELS
        );
        let short = conv
            .store
            .clips
            .iter()
            .find(|c| c.meta["dmax_ms"] == 600)
            .unwrap();
        assert!(cut_view(short, &conv.store.hops(short).unwrap(), 800).is_none());
        let clip = &st.store.clips[st.train[0]];
        let own = st.store.hops(clip).unwrap();
        assert_eq!(
            ext.extended(clip, own.clone(), 1200).unwrap().len(),
            own.len() + 120 * TURN_CHANNELS
        );
        let src = Sources {
            st: &st,
            ext: &ext,
            conv: &conv,
            mix: Mix {
                conv_share: 0.5,
                st_d0: 0.25,
            },
            st_hard: 0.5,
        };
        let (mut order, mut k) = (Order::new(1, st.train.len()), 0u64);
        let (mut conv_rows, mut eot_rows, mut st_rows, mut d0_rows) = (0, 0, 0, 0);
        let mut at5 = None;
        for step in 0..40u64 {
            let (b, rows) = mixed_batch(
                &cfg,
                &src,
                1,
                step,
                32,
                Some(&Augment::PREREG),
                &mut order,
                &mut k,
            )
            .unwrap();
            let hw = b.hw.clone().unwrap();
            for (r, row) in rows.iter().enumerate() {
                let smask = &b.smask[r * TEACHER_STEPS..(r + 1) * TEACHER_STEPS];
                match *row {
                    Row::Conv { record, d } => {
                        conv_rows += 1;
                        eot_rows += conv.store.clips[record].label().unwrap() as usize;
                        assert!(
                            d as u64 <= conv.store.clips[record].meta["dmax_ms"].as_u64().unwrap()
                        );
                        assert_eq!((b.tmask[r], hw[r]), (0.0, 1.0));
                        assert!(smask.iter().all(|&m| m == 0.0));
                    }
                    Row::St { d, .. } => {
                        st_rows += 1;
                        d0_rows += (d == 0) as usize;
                        assert_eq!((b.tmask[r], hw[r]), (1.0, 0.5));
                        // the state term on v2's view only
                        assert_eq!(smask.iter().any(|&m| m > 0.0), d == 0, "d {d}");
                    }
                }
            }
            if step == 5 {
                at5 = Some(b.x.clone());
            }
        }
        let share = conv_rows as f64 / (40.0 * 32.0);
        assert!((share - 0.5).abs() < 0.06, "{share}");
        let bal = eot_rows as f64 / conv_rows as f64;
        assert!((bal - 0.5).abs() < 0.07, "{bal}");
        let d0 = d0_rows as f64 / st_rows as f64;
        assert!((d0 - 0.25).abs() < 0.06, "{d0}");
        // a resume at step 5: the Smart Turn rows take the epoch order where step 4 left it
        let mut order = Order::new(1, st.train.len());
        let mut k = st_draws_before(1, 5, 32, 0.5);
        let (b, _) = mixed_batch(
            &cfg,
            &src,
            1,
            5,
            32,
            Some(&Augment::PREREG),
            &mut order,
            &mut k,
        )
        .unwrap();
        assert_eq!(Some(b.x), at5);
        std::fs::remove_dir_all(&dir).ok();
    }
}
