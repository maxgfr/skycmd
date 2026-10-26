//! s6-cast training data (docs/SPEC-S6-CAST.md §3): windows cut from a pool of synthetic series
//! (refreshed in place as training goes, deterministically in the step) and from real series
//! files, with Reverso's augmentations, missing values and variable context lengths.
//!
//! A window goes through exactly the preparation of inference (`sky_infer::cast::prep`):
//! min-max normalization fitted on the context, right-aligned patches with an observation mask.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
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
}

/// Prepares `context` (its last `context()` points) and `target` (its first `horizon()`). `None`
/// when the context has no observation.
pub fn window(cfg: &CastConfig, context: &[f32], target: &[f32]) -> Option<Window> {
    let norm = Norm::fit(context);
    let (vals, mask) = prep::patches(context, norm, cfg.patch, cfg.context_patches);
    let observed = prep::patch_valid(&mask, cfg.patch);
    let j0 = observed.iter().position(|&v| v)?;
    let valid = (0..cfg.context_patches).map(|j| j >= j0).collect();
    let h = cfg.horizon();
    let mut y = vec![0.0; h];
    let mut ymask = vec![0.0; h];
    for (i, &v) in target.iter().take(h).enumerate() {
        if v.is_finite() {
            y[i] = norm.apply(v);
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
        rows: b,
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
    hash_str(id).is_multiple_of(10)
}

/// `n` synthetic series of `len` points from `seeds`, generated on every core.
fn generate(seeds: &[u64], len: usize) -> Vec<Vec<f32>> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = seeds.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        let handles: Vec<_> = seeds
            .chunks(chunk)
            .map(|c| {
                s.spawn(move || {
                    c.iter()
                        .map(|&sd| synth::mixed(sd, len).1)
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
}

impl Pool {
    pub fn new(seed: u64, size: usize, len: usize, refresh: usize, refresh_every: u64) -> Self {
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
        };
        p.series = generate(&p.seeds_at(0), len);
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
        self.series = generate(&self.seeds_at(step), self.len);
    }

    /// Regenerates the block due at `step`, if any. Returns whether it did.
    pub fn advance(&mut self, step: u64) -> bool {
        if step == 0 || !step.is_multiple_of(self.refresh_every) {
            return false;
        }
        let blocks = (self.size / self.refresh) as u64;
        let g = step / self.refresh_every;
        let o = ((g - 1) % blocks) as usize;
        let seeds: Vec<u64> = (o * self.refresh..(o + 1) * self.refresh)
            .map(|i| self.slot_seed(i, g))
            .collect();
        let fresh = generate(&seeds, self.len);
        for (k, s) in fresh.into_iter().enumerate() {
            self.series[o * self.refresh + k] = s;
        }
        true
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
}

impl Default for SampleOptions {
    fn default() -> Self {
        SampleOptions {
            real_frac: 0.3,
            augment: 0.3,
            mixup: 0.1,
            missing: 0.1,
        }
    }
}

/// Context length of a window, in points: a whole number of patches from 4 to the maximum, more
/// often long than short.
fn context_len(cfg: &CastConfig, rng: &mut Rng) -> usize {
    let c = cfg.context_patches;
    let lo = 4.min(c);
    let k = if rng.f64() < 0.5 {
        c
    } else {
        lo + rng.below(c - lo + 1)
    };
    k * cfg.patch
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
    if ctx < 2 * cfg.patch {
        return None;
    }
    let need = ctx + h;
    let start = srng.below(series.len() - need + 1);
    let mut seg = series[start..start + need].to_vec();
    seg = synth::augment(srng, seg, o.augment, need);
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
pub fn sample(
    cfg: &CastConfig,
    pool: &Pool,
    real: &[Series],
    o: &SampleOptions,
    rng: &mut Rng,
    n: usize,
) -> Vec<Window> {
    let mut out = Vec::with_capacity(n);
    let mut guard = 0;
    while out.len() < n && guard < 100 * n {
        guard += 1;
        let ctx = context_len(cfg, rng);
        let mut srng = SynthRng::new(rng.seed());
        let use_real = !real.is_empty() && rng.f64() < o.real_frac;
        let w = if use_real {
            let s = &real[rng.below(real.len())];
            cut(cfg, &s.values, ctx, &mut srng, o, None)
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
        let ctx_patches = 4 + (i * 7919) % (cfg.context_patches.saturating_sub(4) + 1);
        let ctx = (ctx_patches * cfg.patch).min(cfg.context());
        let (c, t) = y[len - cfg.horizon() - ctx..].split_at(ctx);
        out.extend(window(cfg, c, t));
    }
    for s in real_val {
        let n = s.values.len();
        if n < cfg.horizon() + 2 * cfg.patch {
            continue;
        }
        let (c, t) = s.values.split_at(n - cfg.horizon());
        out.extend(window(cfg, c, t));
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
        for step in 1..=70 {
            a.advance(step);
        }
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
        let draw = |seed| sample(&c, &pool, &real, &o, &mut Rng::new(seed), 32);
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
