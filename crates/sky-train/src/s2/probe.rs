//! `sky-train s2-probe`: closed-loop play of a checkpoint, with or without a memory
//! (`s2::memory`), on the memory probes and on the arena games (docs/s6-sleep-prereg.md §3, §7).
//!
//! The weights are the ternary ones of the checkpoint (fake-quantized as in QAT) and every
//! answer is read at the last exit (τ = 1). A model with a memory starts each episode from a
//! zero memory, writes it at every frame and sleeps `N − 1` passes every `SLEEP_EVERY` frames
//! and at each pause of the game. A model without one answers each frame alone.
//!
//! Two ways to draw the episodes:
//! - probes: one fresh game per episode seed, the seeds being the first ones `≡ 19 (mod 20)`
//!   (the test split of `sky_games::gen::split_for_seed`) from `--from-seed` upward;
//! - arena games: `sky_games::evaluate_with` from `--from-seed`, as `sky-arena --seed` does.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Serialize;
use sky_games::{
    evaluate_with, gen::split_for_seed, make_game, DecisionGame, EvalOptions, Policy, PROBES,
};

use crate::{
    params::{self, ParamMap},
    s2::{
        config::S2Config,
        data::{encode_record_cfg, make_batch, S2Batch, S2Item},
        memory::{sleep, wake, Memory, Sleep, SLEEP_EVERY},
        model::{forward_prepared_mem, logits_per_row, prepare},
    },
    sequence::{qtype_id, DecisionRecord, Tok},
};

pub struct ProbeArgs {
    /// Checkpoint directory (`model.safetensors`, `config.json`, `tokenizer.json`).
    pub run: PathBuf,
    pub tokenizer: Option<PathBuf>,
    pub game: String,
    pub from_seed: u64,
    pub episodes: usize,
    /// Sleep passes to read, one play of the same episodes each.
    pub passes: Vec<usize>,
    /// Arena games: flappy episodes stop after this many pipes (200 in `sky-arena`).
    pub max_pipes: u32,
    /// Read the float weights instead of the ternary ones.
    pub float: bool,
    pub json: Option<PathBuf>,
}

/// One episode of a probe.
#[derive(Debug, Clone, Serialize)]
pub struct Episode {
    pub seed: u64,
    pub score: f32,
    pub max_score: f32,
    /// `recall`: (delay bucket, correct) of each ask.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub asks: Vec<(usize, bool)>,
}

/// The reading of one game at one number of sleep passes.
#[derive(Debug, Clone, Serialize)]
pub struct ProbeReport {
    pub game: String,
    pub memory: bool,
    pub passes: usize,
    pub from_seed: u64,
    pub n_episodes: usize,
    /// Probes: answered questions over asked ones. Arena games: the mean score.
    pub score: f64,
    /// `recall`: accuracy per delay bucket.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub buckets: Vec<f64>,
    /// Probes only.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub episodes: Vec<Episode>,
    /// Arena games only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arena: Option<sky_games::Metrics>,
}

struct MemPolicy<'a> {
    cfg: &'a S2Config,
    p: &'a ParamMap,
    tok: &'a Tok,
    /// (question id, type, instructions) of the game.
    questions: Vec<(String, String, String)>,
    how: Sleep,
    mem: Option<Memory>,
    /// The memory after the current frame's write, kept until the frame ends: a frame with two
    /// questions reads the same memory twice and writes once.
    next: Option<Memory>,
    chunk: Vec<S2Batch>,
    last_steps: u32,
}

impl MemPolicy<'_> {
    fn new_episode(&mut self) {
        self.mem = self
            .cfg
            .memory
            .then(|| Memory::zeros(self.cfg, 1).expect("memory"));
        self.next = None;
        self.chunk.clear();
        self.last_steps = 0;
    }
}

impl Policy for MemPolicy<'_> {
    fn observe(&mut self, g: &dyn DecisionGame) {
        let steps = g.stats().steps;
        if steps < self.last_steps {
            self.new_episode();
        }
        self.last_steps = steps;
        if let Some(m) = self.next.take() {
            self.mem = Some(m);
        }
        if !self.chunk.is_empty() && (self.chunk.len() >= self.how.every || g.pause()) {
            let mem = self.mem.take().expect("a memory model");
            let chunk: Vec<&S2Batch> = self.chunk.iter().collect();
            self.mem = Some(sleep(self.p, self.cfg, &chunk, mem, self.how).expect("sleep"));
            self.chunk.clear();
        }
    }

    fn decide(&mut self, _game: &str, qid: &str, state: &str, opts: &[String]) -> Vec<f32> {
        let (_, t, ins) = self
            .questions
            .iter()
            .find(|(id, _, _)| id == qid)
            .expect("a question of the game")
            .clone();
        let rec = DecisionRecord {
            src: "probe".into(),
            t: t.clone(),
            ins,
            opts: opts.to_vec(),
            state: state.into(),
            target: vec![1.0; opts.len()],
            split: "test".into(),
        };
        let (st, stream) = encode_record_cfg(self.tok, &rec, self.cfg).expect("encode");
        let k = opts.len();
        let item = S2Item {
            stat: 0,
            state: stream,
            qtype: qtype_id(&t).unwrap_or(0),
            target: vec![1.0 / k as f32; k],
            src: "probe".into(),
            group: 0,
        };
        let b = make_batch(std::slice::from_ref(&st), &[&item], 1);
        let n = self.cfg.n_exits();
        let logits = match &self.mem {
            Some(mem) => {
                let (logits, next) = wake(self.p, self.cfg, &b, n, mem).expect("forward");
                if self.next.is_none() {
                    self.next = Some(next);
                    self.chunk.push(b);
                }
                logits
            }
            None => {
                forward_prepared_mem(self.p, self.cfg, &b, n, None, None)
                    .expect("forward")
                    .0
            }
        };
        let per = logits_per_row(&logits, &[k]).expect("logits");
        // Only the argmax is used by the games: the last exit's logits are enough.
        per[n - 1][0].clone()
    }
}

/// The first `n` episode seeds of the test split from `from` upward.
pub fn test_seeds(from: u64, n: usize) -> Vec<u64> {
    let test = |s: &u64| serde_json::to_value(split_for_seed(*s)).is_ok_and(|v| v == "test");
    (from..).filter(test).take(n).collect()
}

fn argmax(v: &[f32]) -> usize {
    (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b })
}

/// Plays one probe episode from its own seed.
fn play_probe(game: &str, seed: u64, pol: &mut MemPolicy<'_>) -> Result<Episode> {
    let mut g = make_game(game, seed).with_context(|| format!("unknown game {game}"))?;
    let q = &g.questions()[0];
    let opts = q.canonical_options();
    pol.new_episode();
    while !g.done() {
        if g.decision_due() {
            pol.observe(&*g);
            let p = pol.decide(game, q.id, &g.state_text(), &opts);
            g.apply(argmax(&p), false);
        } else {
            g.tick();
        }
    }
    let s = g.stats();
    let asks = g.snapshot_json()["asks"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| Some((x["bucket"].as_u64()? as usize, x["correct"].as_bool()?)))
                .collect()
        })
        .unwrap_or_default();
    Ok(Episode {
        seed,
        score: s.score,
        max_score: s.max_score,
        asks,
    })
}

/// Loads a checkpoint as the forward pass uses it (ternary unless `float`).
pub fn load_prepared(run: &Path, float: bool) -> Result<(S2Config, ParamMap)> {
    let cfg = S2Config::load(run.join("config.json"))?;
    let raw = params::load(&run.join("model.safetensors"))?;
    let p = prepare(&raw, &cfg, !float)?.into_owned();
    params::eval(&p)?;
    Ok((cfg, p))
}

/// Plays `a.game` once per value of `a.passes` and prints one line per reading.
pub fn probe(a: &ProbeArgs) -> Result<Vec<ProbeReport>> {
    let (cfg, p) = load_prepared(&a.run, a.float)?;
    let tok_path = a
        .tokenizer
        .clone()
        .unwrap_or_else(|| a.run.join("tokenizer.json"));
    let tok = Tok::load(&tok_path)?;
    let probe_game = PROBES.contains(&a.game.as_str());
    let gm = make_game(&a.game, 0).with_context(|| format!("unknown game {}", a.game))?;
    let mut questions = Vec::new();
    for q in gm.questions() {
        let t = serde_json::to_value(q.qtype)?
            .as_str()
            .unwrap_or("choice")
            .to_string();
        questions.push((q.id.to_string(), t, q.instructions().to_string()));
    }
    // Without a memory, sleep passes change nothing: one reading.
    let passes: Vec<usize> = if cfg.memory {
        a.passes.clone()
    } else {
        vec![1]
    };
    if passes.is_empty() || passes.contains(&0) {
        bail!("--passes needs values >= 1");
    }
    let mut out = Vec::new();
    for &n in &passes {
        let mut pol = MemPolicy {
            cfg: &cfg,
            p: &p,
            tok: &tok,
            questions: questions.clone(),
            how: Sleep {
                passes: n,
                every: SLEEP_EVERY,
                full: true,
            },
            mem: None,
            next: None,
            chunk: Vec::new(),
            last_steps: u32::MAX,
        };
        let report = mlx_rs::with_device(mlx_rs::Device::cpu(), || -> Result<ProbeReport> {
            let mut r = ProbeReport {
                game: a.game.clone(),
                memory: cfg.memory,
                passes: n,
                from_seed: a.from_seed,
                n_episodes: a.episodes,
                score: 0.0,
                buckets: Vec::new(),
                episodes: Vec::new(),
                arena: None,
            };
            if probe_game {
                for seed in test_seeds(a.from_seed, a.episodes) {
                    r.episodes.push(play_probe(&a.game, seed, &mut pol)?);
                }
                let (got, asked) = r.episodes.iter().fold((0.0, 0.0), |(g, m), e| {
                    (g + e.score as f64, m + e.max_score as f64)
                });
                r.score = got / f64::max(asked, 1.0);
                let n_buckets = r
                    .episodes
                    .iter()
                    .flat_map(|e| e.asks.iter().map(|(b, _)| b + 1))
                    .max()
                    .unwrap_or(0);
                for b in 0..n_buckets {
                    let asks: Vec<bool> = r
                        .episodes
                        .iter()
                        .flat_map(|e| e.asks.iter().filter(|(x, _)| *x == b).map(|(_, c)| *c))
                        .collect();
                    let right = asks.iter().filter(|c| **c).count();
                    r.buckets.push(right as f64 / asks.len().max(1) as f64);
                }
            } else {
                let m = evaluate_with(
                    &a.game,
                    &mut pol,
                    a.episodes,
                    a.from_seed,
                    EvalOptions {
                        flappy_max_pipes: a.max_pipes,
                        veto: true,
                    },
                );
                r.score = m.mean_score as f64;
                r.arena = Some(m);
            }
            Ok(r)
        })?;
        let buckets: Vec<String> = report.buckets.iter().map(|x| format!("{x:.3}")).collect();
        println!(
            "{:8} {} N {}: score {:.4} over {} episodes from seed {}{}",
            report.game,
            if report.memory { "memory" } else { "no memory" },
            report.passes,
            report.score,
            report.n_episodes,
            report.from_seed,
            if buckets.is_empty() {
                String::new()
            } else {
                format!(" | by delay {}", buckets.join(" "))
            }
        );
        out.push(report);
    }
    if let Some(path) = &a.json {
        if let Some(d) = path.parent() {
            std::fs::create_dir_all(d)?;
        }
        std::fs::write(path, serde_json::to_string(&out)?)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seeds_are_the_test_split_in_order() {
        let s = test_seeds(1_000_000_000, 3);
        assert_eq!(s, [1_000_000_019, 1_000_000_039, 1_000_000_059]);
        assert_eq!(test_seeds(19, 2), [19, 39]);
    }
}
