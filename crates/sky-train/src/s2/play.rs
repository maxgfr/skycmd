//! `sky-train s2-play`: closed-loop play of an s2/s3 GGUF with the mlx-rs
//! forward pass through the `sky_games` rollout harness (the same metrics as
//! sky-arena), with early exit and, for `temporal_carry` models, the
//! SPEC-S3 §7 latent carried across frames. Reports the mean state block
//! applications per decision on cold frames (no usable carry) and warm frames.
//!
//! The carry of a frame is the raw latent (before the exit norm) after the
//! deepest block application any question of that frame ran; it is used by
//! the next frame when its state token ids are identical, and cleared when a
//! new episode starts.

use std::{collections::HashMap, path::Path};

use anyhow::{Context, Result};
use mlx_rs::Array;
use sky_games::{evaluate_with, make_game, DecisionGame, EvalOptions, Metrics, Policy};
use sky_gguf::Gguf;

use crate::{
    metrics::softmax_t,
    params::ParamMap,
    s2::{
        config::S2Config,
        data::{encode_record_cfg, make_batch, S2Item},
        gguf::{config_from_gguf, params_from_gguf},
        model::{forward_carry, logits_per_row, Carry},
    },
    sequence::{qtype_id, DecisionRecord, Tok},
};

/// Per-game closed-loop result.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PlayReport {
    pub game: String,
    pub tau: f32,
    pub carry: bool,
    pub mean_score: f32,
    pub rescue_rate: f32,
    pub completion_rate: f32,
    pub crashes: u32,
    pub vetoes: u32,
    pub decisions: usize,
    pub mean_layers: f64,
    pub cold_decisions: usize,
    pub cold_mean_layers: f64,
    pub warm_decisions: usize,
    pub warm_mean_layers: f64,
    /// Answers per exit.
    pub exits: Vec<usize>,
}

struct Frame {
    text: String,
    ids: Vec<u32>,
    /// Deepest (exit index, raw latent) run in this frame.
    out: Option<(usize, Array)>,
}

struct MlxPolicy<'a> {
    cfg: &'a S2Config,
    p: &'a ParamMap,
    tok: &'a Tok,
    tau: f32,
    carry: bool,
    questions: HashMap<(String, String), (String, String)>,
    /// Previous frame's carry: (state ids, latent).
    prev: Option<(Vec<u32>, Array)>,
    frame: Option<Frame>,
    last_steps: u32,
    layers: Vec<(bool, usize)>,
    exits: Vec<usize>,
}

impl MlxPolicy<'_> {
    fn end_frame(&mut self) {
        if let Some(f) = self.frame.take() {
            self.prev = f.out.map(|(_, h)| (f.ids, h));
        }
    }
}

impl Policy for MlxPolicy<'_> {
    fn observe(&mut self, g: &dyn DecisionGame) {
        let steps = g.stats().steps;
        if steps < self.last_steps {
            // New episode: no carry across episodes.
            self.frame = None;
            self.prev = None;
        }
        self.last_steps = steps;
    }

    fn decide(&mut self, game: &str, qid: &str, state: &str, opts: &[String]) -> Vec<f32> {
        let (t, ins) = self.questions[&(game.to_string(), qid.to_string())].clone();
        let rec = DecisionRecord {
            src: "play".into(),
            t: t.clone(),
            ins,
            opts: opts.to_vec(),
            state: state.into(),
            target: vec![1.0; opts.len()],
            split: "test".into(),
        };
        let (st, stream) = encode_record_cfg(self.tok, &rec, self.cfg).expect("encode");
        if self.frame.as_ref().is_none_or(|f| f.text != state) {
            self.end_frame();
            self.frame = Some(Frame {
                text: state.into(),
                ids: stream.ids.clone(),
                out: None,
            });
        }
        let k = opts.len();
        let qtype = qtype_id(&t).unwrap_or(0);
        let item = S2Item {
            stat: 0,
            state: stream.clone(),
            qtype,
            target: vec![1.0 / k as f32; k],
            src: "play".into(),
            group: 0,
        };
        let b = make_batch(std::slice::from_ref(&st), &[&item], 1);
        let warm = self.carry
            && self
                .prev
                .as_ref()
                .is_some_and(|(ids, _)| *ids == stream.ids);
        let on = Array::from_slice(&[1.0f32], &[1]);
        let carry = match (&self.prev, warm) {
            (Some((_, h)), true) => Some(Carry { h, on: &on }),
            _ => None,
        };
        let (logits, raw) = forward_carry(self.p, self.cfg, &b, false, carry).expect("forward");
        let per = logits_per_row(&logits, &[k]).expect("logits");
        let n = self.cfg.n_exits();
        let mut chosen = (n - 1, Vec::new());
        for (e, l) in per.iter().enumerate() {
            let q = softmax_t(&l[0], self.cfg.temperature_of(e, qtype, k));
            let top = q.iter().cloned().fold(0.0, f64::max);
            if e + 1 == n || (self.tau < 1.0 && top >= self.tau as f64) {
                chosen = (e, q);
                break;
            }
        }
        let (e, probs) = chosen;
        if self.exits.len() < n {
            self.exits.resize(n, 0);
        }
        self.exits[e] += 1;
        self.layers.push((warm, self.cfg.exits[e]));
        if let Some(f) = &mut self.frame {
            if f.out.as_ref().is_none_or(|(d, _)| e > *d) {
                f.out = Some((e, raw[e].clone()));
            }
        }
        probs.into_iter().map(|x| x as f32).collect()
    }
}

/// Plays `episodes` of every game in closed loop.
#[allow(clippy::too_many_arguments)]
pub fn play(
    gguf: &Path,
    games: &[&str],
    episodes: usize,
    seed: u64,
    tau: f32,
    carry: bool,
    max_pipes: u32,
    tok_path: Option<&Path>,
) -> Result<Vec<PlayReport>> {
    let bytes = std::fs::read(gguf).with_context(|| format!("reading {}", gguf.display()))?;
    let g = Gguf::parse(&bytes)?;
    let cfg = config_from_gguf(&g)?;
    let p = params_from_gguf(&g, &cfg)?;
    let tok_file = match tok_path {
        Some(t) => t.to_path_buf(),
        None => {
            let v = if cfg.vocab_size <= 1024 { 1024 } else { 2048 };
            Path::new("data/tok").join(format!("tok-s2-{v}.json"))
        }
    };
    let tok = Tok::load(&tok_file)?;
    let mut questions = HashMap::new();
    for game in sky_games::GAMES {
        let gm = make_game(game, 0).expect("known game");
        for q in gm.questions() {
            let t = serde_json::to_value(q.qtype)?
                .as_str()
                .unwrap_or("choice")
                .to_string();
            questions.insert(
                (game.to_string(), q.id.to_string()),
                (t, q.instructions().to_string()),
            );
        }
    }
    let mut out = Vec::new();
    for &game in games {
        let mut pol = MlxPolicy {
            cfg: &cfg,
            p: &p,
            tok: &tok,
            tau,
            carry: carry && cfg.temporal_carry,
            questions: questions.clone(),
            prev: None,
            frame: None,
            last_steps: u32::MAX,
            layers: Vec::new(),
            exits: Vec::new(),
        };
        let m: Metrics = mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            evaluate_with(
                game,
                &mut pol,
                episodes,
                seed,
                EvalOptions {
                    flappy_max_pipes: max_pipes,
                    veto: true,
                },
            )
        });
        let mean = |w: Option<bool>| -> (usize, f64) {
            let v: Vec<usize> = pol
                .layers
                .iter()
                .filter(|(x, _)| w.is_none_or(|w| *x == w))
                .map(|(_, l)| *l)
                .collect();
            (
                v.len(),
                v.iter().sum::<usize>() as f64 / v.len().max(1) as f64,
            )
        };
        let (n_all, all) = mean(None);
        let (n_cold, cold) = mean(Some(false));
        let (n_warm, warm) = mean(Some(true));
        let r = PlayReport {
            game: game.into(),
            tau,
            carry: pol.carry,
            mean_score: m.mean_score,
            rescue_rate: m.rescue_rate,
            completion_rate: m.completion_rate,
            crashes: m.crashes,
            vetoes: m.vetoes,
            decisions: n_all,
            mean_layers: all,
            cold_decisions: n_cold,
            cold_mean_layers: cold,
            warm_decisions: n_warm,
            warm_mean_layers: warm,
            exits: pol.exits.clone(),
        };
        println!(
            "{game:7} tau {tau} carry {}: score {:.2} rescue {:.2} completion {:.2} crashes {} vetoes {} | {} decisions, mean layers {:.2} (cold {:.2} on {}, warm {:.2} on {}), exits {:?}",
            r.carry, r.mean_score, r.rescue_rate, r.completion_rate, r.crashes, r.vetoes, r.decisions,
            r.mean_layers, r.cold_mean_layers, r.cold_decisions, r.warm_mean_layers, r.warm_decisions, r.exits
        );
        out.push(r);
    }
    Ok(out)
}
