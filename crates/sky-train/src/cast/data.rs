//! s6-cast training data (docs/SPEC-S6-CAST.md §3): windows cut from a pool of synthetic series
//! (refreshed in place as training goes, deterministically in the step) and from real series
//! files, with Reverso's augmentations, missing values and variable context lengths.
//!
//! A window goes through exactly the preparation of inference (`sky_infer::cast::prep`):
//! min-max normalization fitted on the context, right-aligned patches with an observation mask,
//! and with the v2 front end (docs/s6-cast2-prereg.md §4.2–4.3) the detected periods, the phase
//! channels and the seasonal draft, all from `prep::inputs`, the function the engine calls.
//!
//! v2 also draws its real windows by frequency band (§3.4, [`Bands`]), with the v2 generator
//! periods (C4) and its own sign-flip and time-reversal probability (C5, [`SampleOptions`]).

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use mlx_rs::Array;
use sky_infer::cast::prep::{self, Norm};
use sky_infer::cast::synth::{self, SynthRng};

use super::{config::CastConfig, model::CastBatch};
use crate::rng::{hash_str, Rng};

/// One prepared window.
#[derive(Clone, Debug)]
pub struct Window {
    /// `context()` normalized values and their mask.
    pub vals: Vec<f32>,
    pub mask: Vec<f32>,
    /// Per patch: at or after the first observed patch.
    pub valid: Vec<bool>,
    /// `horizon()` normalized targets (0 where missing) and their mask.
    pub y: Vec<f32>,
    pub ymask: Vec<f32>,
    pub norm: Norm,
    /// Distillation targets: the teacher's normalized quantiles `[horizon][levels]` and their
    /// step mask (all 0 without a teacher, or past the teacher's horizon).
    pub tq: Vec<f32>,
    pub tmask: Vec<f32>,
    /// v2: `[context_patches][2K]` phase channels and `[horizon_patches][draft width]` draft
    /// (empty without the front end), as `prep::inputs` gives them.
    pub phase: Vec<f32>,
    pub draft: Vec<f32>,
    /// Where the window comes from (counted in the log), and the band of its real series.
    pub origin: Origin,
    pub band: Option<Band>,
}

/// The source of a window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    /// The synthetic pool.
    Synth,
    /// A real series.
    Real,
    /// A synthetic series standing in for a band without real series (§3.4).
    BandSynth,
    /// A teacher cut (its targets are the teacher's quantiles too).
    Distilled,
}

impl Origin {
    pub const ALL: [Origin; 4] = [
        Origin::Synth,
        Origin::Real,
        Origin::BandSynth,
        Origin::Distilled,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Origin::Synth => "synthetic",
            Origin::Real => "real",
            Origin::BandSynth => "band_synthetic",
            Origin::Distilled => "distilled",
        }
    }
}

/// Bounds of the normalized targets. A context with almost no range (a flat start followed by a
/// jump) would put targets at 1e3 or more and one window would dominate the clipped gradient of
/// its batch; TinyCast clamps its outputs the same way.
pub const TARGET_MIN: f32 = -5.0;
pub const TARGET_MAX: f32 = 6.0;

/// Prepares `context` (its last `context()` points) and `target` (its first `horizon()`). `None`
/// when the context has no observation.
pub fn window(cfg: &CastConfig, context: &[f32], target: &[f32]) -> Option<Window> {
    let prep::Inputs {
        norm,
        vals,
        mask,
        periods: _,
        phase,
        draft,
    } = prep::inputs(context, &cfg.layout());
    let observed = prep::patch_valid(&mask, cfg.patch);
    let j0 = observed.iter().position(|&v| v)?;
    let valid = (0..cfg.context_patches).map(|j| j >= j0).collect();
    let h = cfg.horizon();
    let mut y = vec![0.0; h];
    let mut ymask = vec![0.0; h];
    for (i, &v) in target.iter().take(h).enumerate() {
        if v.is_finite() {
            y[i] = norm.apply(v).clamp(TARGET_MIN, TARGET_MAX);
            ymask[i] = 1.0;
        }
    }
    Some(Window {
        vals,
        mask,
        valid,
        y,
        ymask,
        norm,
        tq: vec![0.0; h * cfg.quantiles.len()],
        tmask: vec![0.0; h],
        phase,
        draft,
        origin: Origin::Synth,
        band: None,
    })
}

/// The MLX arrays of a batch of windows.
pub fn batch(cfg: &CastConfig, windows: &[Window]) -> CastBatch {
    let b = windows.len();
    let (n, c, h) = (cfg.context(), cfg.context_patches, cfg.horizon());
    let cat = |f: &dyn Fn(&Window) -> &[f32]| -> Vec<f32> {
        windows.iter().flat_map(|w| f(w).iter().copied()).collect()
    };
    let valid: Vec<bool> = windows
        .iter()
        .flat_map(|w| w.valid.iter().copied())
        .collect();
    CastBatch {
        vals: Array::from_slice(&cat(&|w| &w.vals), &[b as i32, n as i32]),
        mask: Array::from_slice(&cat(&|w| &w.mask), &[b as i32, n as i32]),
        valid: Array::from_slice(&valid, &[b as i32, c as i32]),
        y: Array::from_slice(&cat(&|w| &w.y), &[b as i32, h as i32]),
        ymask: Array::from_slice(&cat(&|w| &w.ymask), &[b as i32, h as i32]),
        tq: Array::from_slice(
            &cat(&|w| &w.tq),
            &[b as i32, h as i32, cfg.quantiles.len() as i32],
        ),
        tmask: Array::from_slice(&cat(&|w| &w.tmask), &[b as i32, h as i32]),
        phase: (cfg.periods > 0).then(|| {
            Array::from_slice(
                &cat(&|w| &w.phase),
                &[b as i32, c as i32, 2 * cfg.periods as i32],
            )
        }),
        draft: cfg.draft.then(|| {
            Array::from_slice(
                &cat(&|w| &w.draft),
                &[
                    b as i32,
                    cfg.horizon_patches as i32,
                    cfg.layout().draft_width() as i32,
                ],
            )
        }),
        rows: b,
    }
}

/// A distillation teacher: any `sky_infer::cast::Forecaster` (TinyCast, or an s6-cast file).
pub struct Teacher {
    model: sky_infer::cast::Forecaster,
}

impl Teacher {
    /// Loads a teacher; its quantile levels must be the student's.
    pub fn load(path: &Path, cfg: &CastConfig) -> Result<Self> {
        let mut model = sky_infer::cast::Forecaster::from_bytes(
            &std::fs::read(path).with_context(|| format!("reading {path:?}"))?,
        )?;
        model.prepare_compute(sky_infer::TernaryCompute::Auto);
        if model.levels() != cfg.quantiles.as_slice() {
            anyhow::bail!("the teacher's quantile levels differ from the student's");
        }
        Ok(Teacher { model })
    }

    pub fn name(&self) -> &str {
        self.model.name()
    }

    /// The teacher's raw-unit quantiles of the `h` steps after `context`.
    fn raw(&self, context: &[f32], h: usize) -> Result<Vec<f32>> {
        Ok(self
            .model
            .forecast(context, h, Default::default())?
            .quantiles)
    }

    /// The cached cuts of series `y` (see [`Cut`]), deterministic in `y`'s content and `index`.
    pub fn cuts(&self, cfg: &CastConfig, y: &[f32], index: usize) -> Result<Vec<Cut>> {
        let h = cfg.horizon();
        let Some(room) = y.len().checked_sub(h) else {
            return Ok(Vec::new());
        };
        let ctx = cfg.context().min(room);
        if ctx < SHORT_MIN {
            return Ok(Vec::new());
        }
        let steps = h.min(self.model.pass_horizon());
        let mut rng = Rng::new(hash_str(&format!(
            "cut:{index}:{}:{:x}",
            y.len(),
            y.first().map_or(0, |v| v.to_bits())
        )));
        let mut out = Vec::with_capacity(CUTS_PER_SERIES);
        for _ in 0..CUTS_PER_SERIES {
            let at = ctx + rng.below(room - ctx + 1);
            let context = &y[at - ctx..at];
            if !context.iter().any(|v| v.is_finite()) {
                continue;
            }
            out.push(Cut {
                at,
                ctx,
                steps,
                q: self.raw(context, steps)?,
            });
        }
        Ok(out)
    }
}

/// Teacher cuts per series: a window can only be distilled where the teacher was asked, so the
/// teacher forecasts each series at [`CUTS_PER_SERIES`] fixed points once (when the series enters
/// the pool) instead of on every batch. Computing TinyCast on 128 windows a step held the
/// producer at ~730 windows/s on an M5 instead of ~2,100; cached, a pool refresh costs the
/// teacher 512 forecasts every 50 steps.
pub const CUTS_PER_SERIES: usize = 2;

/// A cached teacher forecast: the window `y[at - ctx .. at]` -> `y[at ..]`, and the teacher's
/// raw quantiles of its first `steps` steps (`[steps][levels]`).
#[derive(Clone, Debug)]
pub struct Cut {
    pub at: usize,
    pub ctx: usize,
    pub steps: usize,
    pub q: Vec<f32>,
}

/// The teacher cuts of the synthetic pool and of the training real series.
pub struct TeacherCache {
    pub pool: Vec<Vec<Cut>>,
    pub real: Vec<Vec<Cut>>,
}

/// `cuts` of every series in `series` (indices `first..`), on `threads` cores.
fn cuts_of(
    cfg: &CastConfig,
    teacher: &Teacher,
    series: &[&[f32]],
    first: usize,
    threads: usize,
) -> Result<Vec<Vec<Cut>>> {
    let per = series.len().div_ceil(threads.max(1)).max(1);
    std::thread::scope(|s| -> Result<Vec<Vec<Cut>>> {
        let handles: Vec<_> = series
            .chunks(per)
            .enumerate()
            .map(|(c, chunk)| {
                s.spawn(move || -> Result<Vec<Vec<Cut>>> {
                    chunk
                        .iter()
                        .enumerate()
                        .map(|(j, y)| teacher.cuts(cfg, y, first + c * per + j))
                        .collect()
                })
            })
            .collect();
        let mut out = Vec::with_capacity(series.len());
        for h in handles {
            out.extend(h.join().expect("teacher worker")?);
        }
        Ok(out)
    })
}

impl TeacherCache {
    pub fn build(
        cfg: &CastConfig,
        teacher: &Teacher,
        pool: &Pool,
        real: &[Series],
        threads: usize,
    ) -> Result<Self> {
        let p: Vec<&[f32]> = pool.series.iter().map(Vec::as_slice).collect();
        let r: Vec<&[f32]> = real.iter().map(|s| s.values.as_slice()).collect();
        Ok(TeacherCache {
            pool: cuts_of(cfg, teacher, &p, 0, threads)?,
            real: cuts_of(cfg, teacher, &r, p.len(), threads)?,
        })
    }

    /// Recomputes the cuts of the pool slots `slots` (after a refresh).
    pub fn refresh(
        &mut self,
        cfg: &CastConfig,
        teacher: &Teacher,
        pool: &Pool,
        slots: std::ops::Range<usize>,
        threads: usize,
    ) -> Result<()> {
        let p: Vec<&[f32]> = pool.series[slots.clone()]
            .iter()
            .map(Vec::as_slice)
            .collect();
        let fresh = cuts_of(cfg, teacher, &p, slots.start, threads)?;
        for (i, c) in slots.zip(fresh) {
            self.pool[i] = c;
        }
        Ok(())
    }

    /// The distilled window of a cut of `y`: the full cut context, no augmentation (the teacher
    /// read the series as it is), the teacher's quantiles normalized like the window.
    pub fn window(cfg: &CastConfig, y: &[f32], c: &Cut) -> Option<Window> {
        let h = cfg.horizon();
        let mut w = window(
            cfg,
            &y[c.at - c.ctx..c.at],
            &y[c.at..(c.at + h).min(y.len())],
        )?;
        let q = cfg.quantiles.len();
        for t in 0..c.steps {
            for k in 0..q {
                w.tq[t * q + k] = w.norm.apply(c.q[t * q + k]).clamp(TARGET_MIN, TARGET_MAX);
            }
            w.tmask[t] = 1.0;
        }
        w.origin = Origin::Distilled;
        Some(w)
    }
}

/// A real series (`{"id": ..., "values": [number | null, ...]}` per JSONL line; `null` is
/// missing).
#[derive(Clone, Debug)]
pub struct Series {
    pub id: String,
    pub values: Vec<f32>,
}

/// Every `*.jsonl` series under `dirs`, in path order.
pub fn load_series(dirs: &[PathBuf]) -> Result<Vec<Series>> {
    let mut files: Vec<PathBuf> = Vec::new();
    for d in dirs {
        if d.is_file() {
            files.push(d.clone());
            continue;
        }
        for e in std::fs::read_dir(d).with_context(|| format!("reading {d:?}"))? {
            let p = e?.path();
            if p.extension().is_some_and(|x| x == "jsonl") {
                files.push(p);
            }
        }
    }
    files.sort();
    let mut out = Vec::new();
    for f in files {
        let text = std::fs::read_to_string(&f).with_context(|| format!("reading {f:?}"))?;
        for (i, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let v: serde_json::Value =
                serde_json::from_str(line).with_context(|| format!("{f:?} line {}", i + 1))?;
            let values = v["values"]
                .as_array()
                .with_context(|| format!("{f:?} line {}: no values", i + 1))?
                .iter()
                .map(|x| x.as_f64().map_or(f32::NAN, |x| x as f32))
                .collect();
            let id = v["id"]
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| format!("{}:{}", f.display(), i + 1));
            out.push(Series { id, values });
        }
    }
    Ok(out)
}

/// Held-out real series: one in ten, by id.
pub fn is_val(id: &str) -> bool {
    hash_str(&val_group(id)).is_multiple_of(10)
}

/// The group a series is held out with: its source (without the sampling rate) and its first
/// id, so the hourly and daily series of one wind farm, the variables of one weather station or
/// of one game episode, stay on the same side of the split.
pub fn val_group(id: &str) -> String {
    let mut parts = id.split(':');
    let source = parts.next().unwrap_or_default();
    let source = ["_hourly", "_daily", "_weekly", "_monthly"]
        .iter()
        .find_map(|s| source.strip_suffix(s))
        .unwrap_or(source);
    match parts.next() {
        Some(first) => format!("{source}:{first}"),
        None => source.to_string(),
    }
}

/// A frequency band of GIFT-Eval (docs/s6-cast2-prereg.md §3.4; `docs/s6-cast2-study.md` §1.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Band {
    Second,
    SubHourly,
    Hourly,
    Daily,
    Weekly,
    Monthly,
}

impl Band {
    pub const ALL: [Band; 6] = [
        Band::Second,
        Band::SubHourly,
        Band::Hourly,
        Band::Daily,
        Band::Weekly,
        Band::Monthly,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Band::Second => "second",
            Band::SubHourly => "subhourly",
            Band::Hourly => "hourly",
            Band::Daily => "daily",
            Band::Weekly => "weekly",
            Band::Monthly => "monthly",
        }
    }

    pub fn parse(s: &str) -> Option<Band> {
        Band::ALL.into_iter().find(|b| b.name() == s)
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// The band of a real series, from its source (the id's first field) and the source's sampling
/// interval. `None` for a source without a calendar frequency (the game trajectories) or not
/// listed here: such a series is not drawn when the bands are on.
pub fn band_of(id: &str) -> Option<Band> {
    let source = id.split(':').next().unwrap_or_default();
    Some(match source {
        // s6-cast v1 (DATA_LICENSES.md): London smart meters every 30 minutes, wind farms,
        // pedestrians and rideshare hourly, Wikipedia and USHCN daily
        "london_smart_meters" => Band::SubHourly,
        "wind_farms_hourly" | "pedestrian_counts" | "rideshare" => Band::Hourly,
        "wind_farms_daily" | "wiki_daily" | "ushcn_daily" => Band::Daily,
        // s6-cast v2 sources (sky-data fetch-series2, DATA_LICENSES.md)
        "uscrn_subhourly" => Band::SubHourly,
        _ => return None,
    })
}

/// Band weights from `"hourly=31,subhourly=30,..."` (every band named at most once; a band left
/// out weighs 0).
pub fn parse_band_weights(s: &str) -> Result<[f64; 6]> {
    let mut w = [0.0; 6];
    let mut seen = [false; 6];
    for part in s.split(',').filter(|p| !p.trim().is_empty()) {
        let (k, v) = part
            .split_once('=')
            .with_context(|| format!("band weight {part:?}: expected band=weight"))?;
        let b = Band::parse(k.trim()).with_context(|| format!("unknown band {k:?}"))?;
        let v: f64 = v
            .trim()
            .parse()
            .with_context(|| format!("band weight {v:?}"))?;
        if !(v >= 0.0 && v.is_finite()) || seen[b.index()] {
            bail!("band weight {part:?}");
        }
        (w[b.index()], seen[b.index()]) = (v, true);
    }
    if w.iter().sum::<f64>() <= 0.0 {
        bail!("the band weights sum to 0");
    }
    Ok(w)
}

/// The periods of the synthetic stand-in of each band, from `"second=360,monthly=12:4"`.
pub fn parse_band_periods(s: &str) -> Result<[Vec<usize>; 6]> {
    let mut out: [Vec<usize>; 6] = Default::default();
    for part in s.split(',').filter(|p| !p.trim().is_empty()) {
        let (k, v) = part
            .split_once('=')
            .with_context(|| format!("band periods {part:?}: expected band=p1:p2"))?;
        let b = Band::parse(k.trim()).with_context(|| format!("unknown band {k:?}"))?;
        let ps = v
            .split(':')
            .map(|p| p.trim().parse::<usize>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .with_context(|| format!("band periods {v:?}"))?;
        if ps.is_empty() || ps.contains(&0) || !out[b.index()].is_empty() {
            bail!("band periods {part:?}");
        }
        out[b.index()] = ps;
    }
    Ok(out)
}

/// The band sampling of real windows (§3.4): a band is drawn with its weight, then a training
/// series of that band, uniformly. A band without real series is stood in for by a synthetic
/// series whose periods are that band's (`synth`).
#[derive(Clone, Debug)]
pub struct Bands {
    pub weights: [f64; 6],
    /// Training series (indices into the real series) per band.
    pub members: [Vec<usize>; 6],
    /// Periods of the synthetic stand-in, per band.
    pub synth: [Vec<usize>; 6],
}

impl Bands {
    /// Every band of positive weight must have real series or stand-in periods.
    pub fn new(real: &[Series], weights: [f64; 6], synth: [Vec<usize>; 6]) -> Result<Self> {
        let mut members: [Vec<usize>; 6] = Default::default();
        for (i, s) in real.iter().enumerate() {
            if let Some(b) = band_of(&s.id) {
                members[b.index()].push(i);
            }
        }
        for b in Band::ALL {
            let i = b.index();
            if weights[i] > 0.0 && members[i].is_empty() && synth[i].is_empty() {
                bail!(
                    "band {} has weight {} but no real series and no stand-in periods (--band-periods)",
                    b.name(),
                    weights[i]
                );
            }
        }
        Ok(Bands {
            weights,
            members,
            synth,
        })
    }

    /// A band by weight, and a series of it (`None`: the band has no real series).
    pub fn draw(&self, rng: &mut Rng) -> (Band, Option<usize>) {
        let b = Band::ALL[rng.weighted(&self.weights)];
        let m = &self.members[b.index()];
        if m.is_empty() {
            (b, None)
        } else {
            (b, Some(m[rng.below(m.len())]))
        }
    }

    /// One line per band: weight, series, stand-in periods.
    pub fn describe(&self, real: &[Series]) -> String {
        let total: f64 = self.weights.iter().sum();
        let mut out = Vec::new();
        for b in Band::ALL {
            let i = b.index();
            let pts: usize = self.members[i].iter().map(|&k| real[k].values.len()).sum();
            out.push(format!(
                "  {:>9}: weight {:.3}, {} series ({pts} points){}",
                b.name(),
                self.weights[i] / total,
                self.members[i].len(),
                if self.members[i].is_empty() && self.weights[i] > 0.0 {
                    format!(", synthetic stand-in at periods {:?}", self.synth[i])
                } else {
                    String::new()
                }
            ));
        }
        let unbanded: std::collections::BTreeSet<&str> = real
            .iter()
            .filter(|s| band_of(&s.id).is_none())
            .map(|s| s.id.split(':').next().unwrap_or_default())
            .collect();
        if !unbanded.is_empty() {
            out.push(format!("  not drawn (no band): {unbanded:?}"));
        }
        out.join("\n")
    }
}

/// `n` synthetic series of `len` points from `seeds`, generated on every core.
fn generate(seeds: &[u64], len: usize, set: &[usize]) -> Vec<Vec<f32>> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = seeds.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        let handles: Vec<_> = seeds
            .chunks(chunk)
            .map(|c| {
                s.spawn(move || {
                    c.iter()
                        .map(|&sd| synth::mixed_in(sd, len, set).1)
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("generator thread"))
            .collect()
    })
}

/// Synthetic pool: `size` series of `len` points. Every `refresh_every` steps a block of
/// `refresh` series is regenerated, cycling over the pool; the content at a step depends only on
/// the step, so a resumed run rebuilds the same pool.
pub struct Pool {
    pub series: Vec<Vec<f32>>,
    size: usize,
    seed: u64,
    len: usize,
    refresh: usize,
    refresh_every: u64,
    /// The generators' period set (`synth::PERIODS`, or `synth::PERIODS_V2`).
    set: &'static [usize],
}

impl Pool {
    /// A pool with the v1 periods.
    pub fn new(seed: u64, size: usize, len: usize, refresh: usize, refresh_every: u64) -> Self {
        Self::new_in(seed, size, len, refresh, refresh_every, &synth::PERIODS)
    }

    /// A pool whose generators draw their periods from `set`.
    pub fn new_in(
        seed: u64,
        size: usize,
        len: usize,
        refresh: usize,
        refresh_every: u64,
        set: &'static [usize],
    ) -> Self {
        assert!(
            refresh > 0 && size.is_multiple_of(refresh),
            "the refresh block must divide the pool"
        );
        let mut p = Pool {
            series: Vec::new(),
            size,
            seed,
            len,
            refresh,
            refresh_every: refresh_every.max(1),
            set,
        };
        p.series = generate(&p.seeds_at(0), len, set);
        p
    }

    fn slot_seed(&self, slot: usize, generation: u64) -> u64 {
        hash_str(&format!("{}:{slot}:{generation}", self.seed))
    }

    /// The seed of every slot at `step`.
    fn seeds_at(&self, step: u64) -> Vec<u64> {
        let n = self.size;
        let blocks = (n / self.refresh) as u64;
        let gens = step / self.refresh_every;
        (0..n)
            .map(|i| {
                let o = (i / self.refresh) as u64;
                // generation g >= 1 rewrites block (g - 1) mod blocks
                let g = if gens > o {
                    o + 1 + (gens - o - 1) / blocks * blocks
                } else {
                    0
                };
                self.slot_seed(i, g)
            })
            .collect()
    }

    /// The pool as it is at `step` (after a resume).
    pub fn rebuild(&mut self, step: u64) {
        self.series = generate(&self.seeds_at(step), self.len, self.set);
    }

    /// Regenerates the block due at `step`, if any, and returns its slots.
    pub fn advance(&mut self, step: u64) -> Option<std::ops::Range<usize>> {
        if step == 0 || !step.is_multiple_of(self.refresh_every) {
            return None;
        }
        let blocks = (self.size / self.refresh) as u64;
        let g = step / self.refresh_every;
        let o = ((g - 1) % blocks) as usize;
        let seeds: Vec<u64> = (o * self.refresh..(o + 1) * self.refresh)
            .map(|i| self.slot_seed(i, g))
            .collect();
        let fresh = generate(&seeds, self.len, self.set);
        for (k, s) in fresh.into_iter().enumerate() {
            self.series[o * self.refresh + k] = s;
        }
        Some(o * self.refresh..(o + 1) * self.refresh)
    }
}

/// Window sampling options.
#[derive(Clone, Copy, Debug)]
pub struct SampleOptions {
    /// Share of windows from real series (when there are any).
    pub real_frac: f64,
    /// Probability of each augmentation.
    pub augment: f64,
    /// Probability of a mixup with a second synthetic series.
    pub mixup: f64,
    /// Probability of dropping 1–5 % of the context values.
    pub missing: f64,
    /// Probability of the sign flip and of the time reversal (v1: `augment`; v2, C5: 0.5).
    pub flip: f64,
    /// The periods of the synthetic stand-ins' generators, and of the pool's.
    pub set: &'static [usize],
}

impl Default for SampleOptions {
    fn default() -> Self {
        SampleOptions {
            real_frac: 0.3,
            augment: 0.3,
            mixup: 0.1,
            missing: 0.1,
            flip: 0.3,
            set: &synth::PERIODS,
        }
    }
}

/// Context length of a window, in points: a whole number of patches from 4 to the maximum, more
/// often long than short.
/// Context lengths, in points: the full context half of the time, a whole number of patches
/// from 4 up 30 % of the time, and 8–256 points (log-uniform, any length, so a partly filled
/// first patch) the other 20 % — GIFT-Eval's short configurations (yearly or quarterly M4, car
/// parts, hospital) give the model 13 to 300 points.
const SHORT_SHARE: f64 = 0.2;
const SHORT_MIN: usize = 8;
const SHORT_MAX: usize = 256;

fn context_len(cfg: &CastConfig, rng: &mut Rng) -> usize {
    let c = cfg.context_patches;
    let r = rng.f64();
    if r < SHORT_SHARE {
        let (lo, hi) = (SHORT_MIN as f64, SHORT_MAX.min(cfg.context()) as f64);
        return (lo * (hi / lo).powf(rng.f64())).round() as usize;
    }
    if r < 0.5 {
        let lo = 4.min(c);
        return (lo + rng.below(c - lo + 1)) * cfg.patch;
    }
    cfg.context()
}

/// A window from `series` (`need` points or more), with the augmentations.
fn cut(
    cfg: &CastConfig,
    series: &[f32],
    ctx: usize,
    srng: &mut SynthRng,
    o: &SampleOptions,
    mix: Option<&[f32]>,
) -> Option<Window> {
    let h = cfg.horizon();
    let ctx = ctx.min(series.len().checked_sub(h)?);
    if ctx < SHORT_MIN {
        return None;
    }
    let need = ctx + h;
    // downsampling: cut k times as many points and keep every k-th, when the series has them
    let k = if srng.coin(o.augment) {
        [2, 4]
            .into_iter()
            .rev()
            .find(|&k| series.len() >= need * k && srng.coin(0.5))
            .unwrap_or(1)
    } else {
        1
    };
    let span = need * k;
    let start = srng.below(series.len() - span + 1);
    let mut seg = synth::downsample(&series[start..start + span], k);
    seg.truncate(need);
    seg = synth::augment_with(srng, seg, o.augment, o.flip);
    if let Some(m) = mix {
        if m.len() >= need && srng.coin(o.mixup) {
            let s = srng.below(m.len() - need + 1);
            seg = synth::mixup(srng, &seg, &m[s..s + need]);
        }
    }
    if srng.coin(o.missing) {
        let frac = srng.range(0.01, 0.05);
        for v in seg[..ctx].iter_mut() {
            if srng.coin(frac) {
                *v = f32::NAN;
            }
        }
    }
    let (c, t) = seg.split_at(ctx);
    window(cfg, c, t)
}

/// Samples `n` training windows.
/// The windows of training step `step`: sampled with a generator seeded from `(seed, step)`
/// only, so a resumed run (or a prefetching thread) draws exactly the same batches. A share
/// `teacher_frac` of them gets the teacher's targets.
#[allow(clippy::too_many_arguments)]
pub fn sample_step(
    cfg: &CastConfig,
    pool: &Pool,
    real: &[Series],
    o: &SampleOptions,
    seed: u64,
    step: u64,
    n: usize,
    teacher: Option<(&TeacherCache, f64)>,
    bands: Option<&Bands>,
) -> Result<Vec<Window>> {
    let mut rng = Rng::new(hash_str(&format!("cast-step:{seed}:{step}")));
    let mut windows = sample(cfg, pool, real, o, &mut rng, n, bands);
    if let Some((cache, frac)) = teacher {
        // a share `frac` of the windows is replaced by distilled windows from the teacher cuts
        for w in windows.iter_mut() {
            if rng.f64() >= frac {
                continue;
            }
            // the real series of a distilled window: uniform (v1), or by band (v2), where a band
            // without real series gives a pool cut (the stand-ins have no teacher cut)
            let real_index = match bands {
                None => (!cache.real.is_empty() && rng.f64() < o.real_frac)
                    .then(|| rng.below(cache.real.len())),
                Some(b) => {
                    if !cache.real.is_empty() && rng.f64() < o.real_frac {
                        b.draw(&mut rng).1
                    } else {
                        None
                    }
                }
            };
            let (cuts, y): (&[Cut], &[f32]) = if let Some(i) = real_index {
                (&cache.real[i], &real[i].values)
            } else {
                let i = rng.below(cache.pool.len());
                (&cache.pool[i], &pool.series[i])
            };
            if cuts.is_empty() {
                continue;
            }
            if let Some(mut d) = TeacherCache::window(cfg, y, &cuts[rng.below(cuts.len())]) {
                d.band = real_index.and_then(|i| band_of(&real[i].id));
                *w = d;
            }
        }
    }
    Ok(windows)
}

pub fn sample(
    cfg: &CastConfig,
    pool: &Pool,
    real: &[Series],
    o: &SampleOptions,
    rng: &mut Rng,
    n: usize,
    bands: Option<&Bands>,
) -> Vec<Window> {
    let mut out = Vec::with_capacity(n);
    let mut guard = 0;
    // the stand-in series of a band without real series: as long as the pool's v1 series
    let stand_in_len = cfg.context() + cfg.horizon() + 1024;
    while out.len() < n && guard < 100 * n {
        guard += 1;
        let ctx = context_len(cfg, rng);
        let mut srng = SynthRng::new(rng.seed());
        if let Some(b) = bands {
            // v2 (§3.4): a real window with probability real_frac, its band drawn by weight
            if rng.f64() < o.real_frac {
                let (band, idx) = b.draw(rng);
                let w = match idx {
                    Some(i) => cut(cfg, &real[i].values, ctx, &mut srng, o, None).map(|mut w| {
                        (w.origin, w.band) = (Origin::Real, Some(band));
                        w
                    }),
                    None => {
                        let (_, y) =
                            synth::mixed_in(rng.seed(), stand_in_len, &b.synth[band.index()]);
                        cut(cfg, &y, ctx, &mut srng, o, None).map(|mut w| {
                            (w.origin, w.band) = (Origin::BandSynth, Some(band));
                            w
                        })
                    }
                };
                out.extend(w);
                continue;
            }
            let s = &pool.series[rng.below(pool.series.len())];
            let m = &pool.series[rng.below(pool.series.len())];
            out.extend(cut(cfg, s, ctx, &mut srng, o, Some(m)));
            continue;
        }
        let use_real = !real.is_empty() && rng.f64() < o.real_frac;
        let w = if use_real {
            let s = &real[rng.below(real.len())];
            cut(cfg, &s.values, ctx, &mut srng, o, None).map(|mut w| {
                w.origin = Origin::Real;
                w
            })
        } else {
            let s = &pool.series[rng.below(pool.series.len())];
            let m = &pool.series[rng.below(pool.series.len())];
            cut(cfg, s, ctx, &mut srng, o, Some(m))
        };
        if let Some(w) = w {
            out.push(w);
        }
    }
    out
}

/// The fixed validation windows: `n_synth` synthetic series (seeds unseen in training) cut at
/// varied context lengths, then the last window of every held-out real series.
pub fn validation(cfg: &CastConfig, n_synth: usize, real_val: &[Series]) -> Vec<Window> {
    let mut out = Vec::new();
    let len = cfg.context() + cfg.horizon();
    for i in 0..n_synth {
        let (_, y) = synth::mixed(hash_str(&format!("val:{i}")), len);
        // one in five short (8–256 points, any length), as in training
        let ctx = if i % 5 == 0 {
            SHORT_MIN + (i * 7919) % (SHORT_MAX.min(cfg.context()) - SHORT_MIN + 1)
        } else {
            let ctx_patches = 4 + (i * 7919) % (cfg.context_patches.saturating_sub(4) + 1);
            (ctx_patches * cfg.patch).min(cfg.context())
        };
        let (c, t) = y[len - cfg.horizon() - ctx..].split_at(ctx);
        out.extend(window(cfg, c, t));
    }
    for s in real_val {
        let n = s.values.len();
        if n < cfg.horizon() + SHORT_MIN {
            continue;
        }
        let (c, t) = s.values.split_at(n - cfg.horizon());
        out.extend(window(cfg, c, t).map(|mut w| {
            (w.origin, w.band) = (Origin::Real, band_of(&s.id));
            w
        }));
    }
    out
}

/// The JSONL line of a series (for `cast-synth`).
pub fn series_line(id: &str, values: &[f32]) -> String {
    let vals: Vec<serde_json::Value> = values
        .iter()
        .map(|v| {
            if v.is_finite() {
                serde_json::json!(v)
            } else {
                serde_json::Value::Null
            }
        })
        .collect();
    serde_json::json!({"id": id, "values": vals}).to_string()
}

/// Writes `n` synthetic series of `len` points (family `kind`, or mixed) to `out` as JSONL.
pub fn write_synth(
    out: &Path,
    kind: Option<synth::Kind>,
    n: usize,
    len: usize,
    seed: u64,
) -> Result<()> {
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut text = String::new();
    for i in 0..n {
        let s = hash_str(&format!("{seed}:{i}"));
        let (k, y) = match kind {
            Some(k) => (k, synth::series(k, s, len)),
            None => synth::mixed(s, len),
        };
        text.push_str(&series_line(&format!("{}-{i}", k.name()), &y));
        text.push('\n');
    }
    std::fs::write(out, text)?;
    Ok(())
}

/// The numbers of a game's state text, by key: `"vy -2 dx 35 t N"` gives `vy` → −2 and
/// `dx` → 35 (a key followed by a non-number is skipped).
pub fn state_numbers(text: &str) -> Vec<(String, f32)> {
    let toks: Vec<&str> = text.split_whitespace().collect();
    toks.windows(2)
        .filter_map(|w| {
            let key_like = w[0].parse::<f32>().is_err();
            let v = w[1].parse::<f32>().ok()?;
            (key_like && v.is_finite()).then(|| (w[0].to_string(), v))
        })
        .collect()
}

/// Series of the state variables of `episodes` episodes of `game`, played by the oracle with
/// 10 % random decisions (so the trajectories are not all the oracle's): one series per state key
/// and episode, sampled at every decision. Constant or short series are dropped.
pub fn game_series(
    game: &str,
    episodes: usize,
    seed: u64,
    max_steps: usize,
) -> Result<Vec<Series>> {
    let mut g =
        sky_games::make_game(game, seed).with_context(|| format!("unknown game {game:?}"))?;
    let mut rng = Rng::new(seed ^ 0x6a3e);
    let mut out = Vec::new();
    for ep in 0..episodes {
        g.reset();
        let mut keys: Vec<String> = Vec::new();
        let mut cols: Vec<Vec<f32>> = Vec::new();
        let mut frames = 0usize;
        let mut steps = 0usize;
        while !g.done() && steps < max_steps {
            steps += 1;
            if !g.decision_due() {
                g.tick();
                continue;
            }
            for (k, v) in state_numbers(&g.state_text()) {
                let i = match keys.iter().position(|x| *x == k) {
                    Some(i) => i,
                    None => {
                        keys.push(k);
                        cols.push(vec![f32::NAN; frames]);
                        keys.len() - 1
                    }
                };
                cols[i].push(v);
            }
            frames += 1;
            for c in cols.iter_mut() {
                c.resize(frames, f32::NAN);
            }
            let oracle = g.oracle();
            let n = oracle[0].len();
            let choice = if rng.f64() < 0.1 {
                rng.below(n)
            } else {
                sky_games::common::argmax(&oracle[0])
            };
            g.apply(choice, false);
        }
        for (k, v) in keys.iter().zip(cols) {
            let finite: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
            let varies = finite.windows(2).any(|w| w[0] != w[1]);
            if finite.len() >= 64 && varies {
                out.push(Series {
                    id: format!("{game}:{ep}:{k}"),
                    values: v,
                });
            }
        }
    }
    Ok(out)
}

/// Writes the game series of `games` to `out` as JSONL.
pub fn write_games(out: &Path, games: &[String], episodes: usize, seed: u64) -> Result<usize> {
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut text = String::new();
    let mut n = 0;
    for game in games {
        for s in game_series(game, episodes, seed, 20_000)? {
            text.push_str(&series_line(&s.id, &s.values));
            text.push('\n');
            n += 1;
        }
    }
    std::fs::write(out, text)?;
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> CastConfig {
        CastConfig::load(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/s6-cast-femto.json"),
        )
        .unwrap()
    }

    fn cfg2() -> CastConfig {
        CastConfig::load(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/s6-cast2-femto.json"),
        )
        .unwrap()
    }

    /// Contexts with a known period, with none, a short history, missing values, and longer
    /// than the context.
    fn fixtures() -> Vec<Vec<f32>> {
        let tau = 2.0 * std::f32::consts::PI;
        let mut gaps: Vec<f32> = (0..200).map(|t| (tau * t as f32 / 12.0).sin()).collect();
        for (i, v) in gaps.iter_mut().enumerate() {
            if i % 9 == 4 || (60..75).contains(&i) {
                *v = f32::NAN;
            }
        }
        vec![
            (0..400)
                .map(|t| 3.0 + (tau * t as f32 / 24.0).sin())
                .collect(),
            vec![1.5; 120],
            (0..29).map(|t| (tau * t as f32 / 6.0).cos()).collect(),
            gaps,
            (0..1500)
                .map(|t| t as f32 * 0.1 + (t as f32 * 0.7).sin())
                .collect(),
            synth::series(synth::Kind::Sarima, 3, 333),
        ]
    }

    /// The trainer's front end is the engine's: the window of a context holds exactly the
    /// numbers `sky_infer::cast::CastModel` computes for it (`prep::inputs` with the layout the
    /// engine reads from the GGUF keys), bit for bit. CPU only: no MLX array is made.
    #[test]
    fn front_end_parity_with_the_engine() {
        for c in [cfg2(), cfg()] {
            let engine = c.infer();
            assert_eq!(engine.layout(), c.layout());
            engine.validate().unwrap();
            for ctx in fixtures() {
                let w = window(&c, &ctx, &[]).unwrap();
                let i = prep::inputs(&ctx, &engine.layout());
                assert_eq!(w.norm, i.norm);
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&w.vals), bits(&i.vals));
                assert_eq!(bits(&w.mask), bits(&i.mask));
                assert_eq!(bits(&w.phase), bits(&i.phase));
                assert_eq!(bits(&w.draft), bits(&i.draft));
                let l = c.layout();
                assert_eq!(w.phase.len(), c.context_patches * l.phase_width());
                assert_eq!(w.draft.len(), c.horizon_patches * l.draft_width());
                if c.periods > 0 {
                    assert_eq!(i.periods.len(), c.periods);
                }
            }
        }
        // the fixtures hold a detected period and a context without one
        let l = cfg2().layout();
        let found: Vec<usize> = fixtures()
            .iter()
            .map(|x| prep::inputs(x, &l).periods[0])
            .collect();
        assert_eq!(found[0], 24, "{found:?}");
        assert_eq!(found[1], 0, "{found:?}");
    }

    /// The batch tensors of the front end: `[B, C, 2K]` phase channels and `[B, hq, width]`
    /// drafts in window order (MLX on the CPU device).
    #[test]
    fn batch_carries_the_front_end() {
        let _g = crate::test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            let c = cfg2();
            let ws: Vec<Window> = fixtures()
                .iter()
                .map(|x| window(&c, x, &[0.0; 4]).unwrap())
                .collect();
            let b = batch(&c, &ws);
            let ph = b.phase.as_ref().unwrap();
            let dr = b.draft.as_ref().unwrap();
            assert_eq!(ph.shape(), &[ws.len() as i32, 40, 8]);
            assert_eq!(dr.shape(), &[ws.len() as i32, 2, 24]);
            let flat: Vec<f32> = ws.iter().flat_map(|w| w.draft.clone()).collect();
            assert_eq!(dr.as_slice::<f32>(), flat.as_slice());
            assert!(batch(&cfg(), &ws[..1]).phase.is_none());
        });
    }

    #[test]
    fn bands_parse_and_draw() {
        let w = parse_band_weights("hourly=31,subhourly=30,daily=15,weekly=8,monthly=7,second=6")
            .unwrap();
        assert_eq!(w, [6.0, 30.0, 31.0, 15.0, 8.0, 7.0]);
        assert!(parse_band_weights("hourly=1,hourly=2").is_err());
        assert!(parse_band_weights("yearly=1").is_err());
        let p = parse_band_periods("second=360,monthly=12:4").unwrap();
        assert_eq!((p[0].clone(), p[5].clone()), (vec![360], vec![12, 4]));
        let real: Vec<Series> = ["london_smart_meters:T1", "wiki_daily:T2", "flappy:0:vy"]
            .iter()
            .map(|id| Series {
                id: id.to_string(),
                values: vec![1.0; 500],
            })
            .collect();
        assert_eq!(band_of("flappy:0:vy"), None);
        // a band of positive weight with neither real series nor stand-in periods is an error
        assert!(Bands::new(&real, w, Default::default()).is_err());
        let synth_p = parse_band_periods("second=360,hourly=24,weekly=52,monthly=12:4").unwrap();
        let b = Bands::new(&real, w, synth_p).unwrap();
        assert_eq!(b.members[Band::SubHourly as usize], vec![0]);
        assert_eq!(b.members[Band::Daily as usize], vec![1]);
        let mut rng = Rng::new(1);
        let mut hits = [0usize; 6];
        let mut game = 0;
        for _ in 0..9700 {
            let (band, i) = b.draw(&mut rng);
            hits[band as usize] += 1;
            game += usize::from(i == Some(2));
        }
        assert_eq!(game, 0, "the game series has no band");
        // about 31 / 97 hourly, 6 / 97 second-level
        assert!(
            (2700..3500).contains(&hits[Band::Hourly as usize]),
            "{hits:?}"
        );
        assert!(
            (400..800).contains(&hits[Band::Second as usize]),
            "{hits:?}"
        );
        // the windows: real ones from their band, stand-ins for the empty bands
        let c = cfg2();
        let pool = Pool::new_in(4, 8, 6048, 2, 50, &synth::PERIODS_V2);
        let o = SampleOptions {
            real_frac: 0.6,
            flip: 0.5,
            set: &synth::PERIODS_V2,
            ..SampleOptions::default()
        };
        let ws = sample(&c, &pool, &real, &o, &mut Rng::new(2), 300, Some(&b));
        let count = |o: Origin| ws.iter().filter(|w| w.origin == o).count();
        assert!(
            count(Origin::Real) > 0 && count(Origin::BandSynth) > 0 && count(Origin::Synth) > 0
        );
        assert!(ws
            .iter()
            .filter(|w| w.origin == Origin::Real)
            .all(|w| matches!(w.band, Some(Band::SubHourly | Band::Daily))));
        assert!(ws
            .iter()
            .all(|w| w.phase.len() == 40 * 8 && w.draft.len() == 2 * 24));
    }

    #[test]
    fn game_state_texts_become_series() {
        assert_eq!(
            state_numbers("vy -2 dx 35 t N alt 32.5"),
            vec![
                ("vy".into(), -2.0),
                ("dx".into(), 35.0),
                ("alt".into(), 32.5)
            ]
        );
        let s = game_series("flappy", 2, 1, 4000).unwrap();
        assert!(!s.is_empty());
        assert!(s
            .iter()
            .all(|x| x.id.starts_with("flappy:") && x.values.len() >= 64));
        assert!(game_series("nope", 1, 1, 10).is_err());
    }

    #[test]
    fn teacher_cuts_from_tinycast() {
        let c = cfg();
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../sky-infer/tests/golden/tinycast.gguf");
        let t = Teacher::load(&path, &c).unwrap();
        let y: Vec<f32> = (0..600).map(|i| 10.0 + (i as f32 * 0.3).sin()).collect();
        let cuts = t.cuts(&c, &y, 0).unwrap();
        assert_eq!(cuts.len(), CUTS_PER_SERIES);
        // deterministic in the content
        assert_eq!(t.cuts(&c, &y, 0).unwrap()[0].at, cuts[0].at);
        let w = TeacherCache::window(&c, &y, &cuts[0]).unwrap();
        // femto forecasts 16 steps, within TinyCast's 48-step pass: every step is covered
        assert!(w.tmask.iter().all(|&m| m == 1.0));
        let q = c.quantiles.len();
        for row in w.tq.chunks(q) {
            assert!(row.windows(2).all(|x| x[0] <= x[1]) && row.iter().all(|v| v.is_finite()));
        }
        // a sine in [9, 11] normalizes to [0, 1]: the median stays in range
        assert!(w
            .tq
            .iter()
            .skip(q / 2)
            .step_by(q)
            .all(|&v| (-0.5..1.5).contains(&v)));
        // too short a series has no cut
        assert!(t.cuts(&c, &y[..10], 1).unwrap().is_empty());
        // a distilled share of the step's windows comes from the cache
        let pool = Pool::new(3, 8, c.context() + c.horizon() + 64, 2, 50);
        let cache = TeacherCache::build(&c, &t, &pool, &[], 2).unwrap();
        let ws = sample_step(
            &c,
            &pool,
            &[],
            &SampleOptions::default(),
            1,
            0,
            40,
            Some((&cache, 0.5)),
            None,
        )
        .unwrap();
        let distilled = ws.iter().filter(|w| w.tmask[0] == 1.0).count();
        assert!((5..=35).contains(&distilled), "{distilled} of 40");
    }

    #[test]
    fn validation_groups_and_bounded_targets() {
        // the hourly and daily series of one farm, the variables of one station, one episode
        assert_eq!(
            val_group("wind_farms_hourly:T1"),
            val_group("wind_farms_daily:T1")
        );
        assert_eq!(
            val_group("ushcn_daily:T9:TMAX"),
            val_group("ushcn_daily:T9:PRCP")
        );
        assert_eq!(val_group("flappy:3:vy"), val_group("flappy:3:alt"));
        assert_ne!(val_group("flappy:3:vy"), val_group("flappy:4:vy"));
        assert_eq!(
            is_val("wind_farms_hourly:T1"),
            is_val("wind_farms_daily:T1")
        );
        // a flat context then a jump: the targets are clamped, not 1e6
        let c = cfg();
        let w = window(&c, &[1.0; 40], &[1e6; 16]).unwrap();
        assert!(w.y.iter().all(|&v| v == TARGET_MAX));
    }

    #[test]
    fn short_contexts_are_sampled() {
        let c = cfg();
        let pool = Pool::new(2, 16, c.context() + c.horizon() + 64, 4, 50);
        let ws = sample(
            &c,
            &pool,
            &[],
            &SampleOptions::default(),
            &mut Rng::new(5),
            400,
            None,
        );
        let observed = |w: &Window| w.mask.iter().filter(|&&m| m > 0.0).count();
        // some windows read fewer than 4 patches (8–31 points), so their first patch is partial
        let short = ws.iter().filter(|w| observed(w) < 4 * c.patch).count();
        assert!(short > 20, "{short} short windows of 400");
        // and every window still holds at least SHORT_MIN observed points
        assert!(ws.iter().all(|w| observed(w) >= SHORT_MIN.min(8) / 2));
    }

    #[test]
    fn windows_mask_left_padding_and_normalize_targets() {
        let c = cfg();
        let ctx: Vec<f32> = (0..20).map(|t| t as f32).collect();
        let tgt = [20.0, f32::NAN, 22.0];
        let w = window(&c, &ctx, &tgt).unwrap();
        assert_eq!(w.vals.len(), c.context());
        // 20 points in patches of 8: the last 3 patches hold data (4 + 8 + 8)
        let first = c.context_patches - 3;
        assert!(w.valid[first] && !w.valid[first - 1]);
        assert_eq!(
            w.norm,
            Norm {
                lo: 0.0,
                scale: 19.0
            }
        );
        assert!((w.y[0] - 20.0 / 19.0).abs() < 1e-6);
        assert_eq!(&w.ymask[..3], &[1.0, 0.0, 1.0]);
        assert!(window(&c, &[f32::NAN; 4], &tgt).is_none());
    }

    #[test]
    fn pool_refresh_is_a_function_of_the_step() {
        let mut a = Pool::new(5, 8, 64, 2, 10);
        let start = a.series.clone();
        // the refreshed slots, for the teacher cache: block g % 4 at generation g
        let slots: Vec<_> = (1..=70).filter_map(|step| a.advance(step)).collect();
        assert_eq!(slots, [0..2, 2..4, 4..6, 6..8, 0..2, 2..4, 4..6]);
        assert_ne!(a.series, start);
        let mut b = Pool::new(5, 8, 64, 2, 10);
        b.rebuild(70);
        assert_eq!(a.series, b.series);
        // seven refreshes of 2 out of 8: 4 blocks, block 0 rewritten at generations 1 and 5
        let mut c = Pool::new(5, 8, 64, 2, 10);
        c.rebuild(0);
        assert_eq!(c.series, start);
    }

    #[test]
    fn sampling_is_deterministic_and_well_formed() {
        let c = cfg();
        let pool = Pool::new(1, 16, c.context() + c.horizon() + 64, 4, 50);
        let real = vec![Series {
            id: "a".into(),
            values: (0..400).map(|t| (t as f32 * 0.3).sin()).collect(),
        }];
        let o = SampleOptions::default();
        let draw = |seed| sample(&c, &pool, &real, &o, &mut Rng::new(seed), 32, None);
        let (x, y) = (draw(3), draw(3));
        assert_eq!(x.len(), 32);
        assert_eq!(
            x.iter().map(|w| w.vals.clone()).collect::<Vec<_>>(),
            y.iter().map(|w| w.vals.clone()).collect::<Vec<_>>()
        );
        for w in &x {
            assert!(w.vals.iter().all(|v| v.is_finite()));
            assert!(w.valid.iter().any(|&v| v));
            assert_eq!(w.y.len(), c.horizon());
        }
        let v = validation(&c, 10, &real);
        assert_eq!(v.len(), 11);
    }
}
