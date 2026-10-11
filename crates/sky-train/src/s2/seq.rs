//! SPEC-S3 §7 temporal-recursion data: `sky-train s2-seqgen` plays the games
//! through the public `sky_games::DecisionGame` API (oracle play with the
//! generator's epsilon noise) and emits sequences of `K` consecutive decision
//! points of one episode, each frame with its oracle target, all frames of a
//! sequence asking the same rendered question. Every record is a regular
//! decision record plus `seq` (sequence id) and `seq_t` (frame index; `t` is
//! already the question type of the record schema).

use std::{
    collections::HashMap,
    io::{BufRead, BufReader, Write},
    path::Path,
};

use anyhow::{bail, Context, Result};
use rand::{rngs::SmallRng, Rng as _, SeedableRng};
use serde_json::json;
use sky_games::{gen::render_variant, gen::split_for_seed, make_game};

use crate::{
    decision::list_jsonl,
    rng::Rng,
    s2::{
        config::S2Config,
        data::{self, make_batch_min, S2Batch, S2Data},
    },
    sequence::Tok,
};

/// Noise levels of the episodes (as the sky-games generator).
const EPSILONS: [f32; 4] = [0.0, 0.05, 0.15, 0.3];
/// Share of sequences asking the secondary question of a game.
const SECONDARY_SHARE: f64 = 0.3;

fn argmax(v: &[f32]) -> usize {
    (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b })
}

/// Sequence id, unique across games (FNV-1a of the game name in the top 16
/// bits), seeds and sequence numbers.
pub fn seq_id(game: &str, seed: u64, n: usize) -> u64 {
    let h = game.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x0100_0000_01b3)
    });
    (h & 0xffff) << 48 | (seed & 0xffff) << 32 | (n as u64 & 0xffff_ffff)
}

/// Writes `n_seq` sequences of `k` frames for `game` to `out` (JSONL);
/// returns the number of episodes played.
pub fn generate(game: &str, n_seq: usize, k: usize, seed: u64, out: &Path) -> Result<u64> {
    let probe = make_game(game, 0).with_context(|| format!("unknown game {game}"))?;
    let questions = probe.questions();
    let mut rng = SmallRng::seed_from_u64(seed ^ 0x7365_7167);
    if let Some(d) = out.parent() {
        std::fs::create_dir_all(d)?;
    }
    let mut w = std::io::BufWriter::new(std::fs::File::create(out)?);
    let (mut emitted, mut ep) = (0usize, 0u64);
    let max_score = if game == "flappy" { 12.0 } else { f32::MAX };
    while emitted < n_seq {
        let episode_seed = seed.wrapping_mul(1_000_000).wrapping_add(ep);
        ep += 1;
        let split = serde_json::to_value(split_for_seed(episode_seed))?;
        let eps = EPSILONS[rng.gen_range(0..EPSILONS.len())];
        let mut g = make_game(game, episode_seed).expect("known game");
        // (state text, oracle targets) of the recent decision points.
        let mut window: Vec<(String, Vec<Vec<f32>>)> = Vec::new();
        while !g.done() && g.stats().score < max_score && emitted < n_seq {
            if !g.decision_due() {
                g.tick();
                continue;
            }
            let t = g.oracle();
            window.push((g.state_text(), t.clone()));
            if window.len() == k {
                // Non-overlapping windows, half of them kept, so sequences
                // spread over the whole episode.
                if rng.gen_bool(0.5) {
                    let qi = if questions.len() > 1 && rng.gen_bool(SECONDARY_SHARE) {
                        1 + rng.gen_range(0..questions.len() - 1)
                    } else {
                        0
                    };
                    let q = &questions[qi];
                    let r = render_variant(q, &mut rng);
                    let qtype = serde_json::to_value(q.qtype)?;
                    let id = seq_id(game, seed, emitted);
                    for (ft, (state, targets)) in window.iter().enumerate() {
                        let rec = json!({
                            "src": format!("seq:{game}"), "t": qtype, "ins": r.ins, "opts": r.opts,
                            "state": state, "target": r.map_target(&targets[qi]), "split": split,
                            "seq": id, "seq_t": ft,
                        });
                        serde_json::to_writer(&mut w, &rec)?;
                        w.write_all(b"\n")?;
                    }
                    emitted += 1;
                }
                window.clear();
            }
            let n_opts = questions[0].options.len();
            let choice = if rng.gen::<f32>() < eps {
                rng.gen_range(0..n_opts)
            } else {
                argmax(&t[0])
            };
            let abort = game == "dodge" && t.len() > 1 && t[1][1] > 0.5 && rng.gen_bool(0.3);
            g.apply(choice, abort);
        }
    }
    w.flush()?;
    Ok(ep)
}

/// Sequence data: the decision items plus the train sequences (indices into
/// `data.train[0]`, `k` frames each, in frame order).
pub struct SeqData {
    pub data: S2Data,
    pub train: Vec<Vec<usize>>,
    pub k: usize,
}

/// Loads `s2-seqgen` output (a file or a directory) with the stream settings of `cfg`.
pub fn load(path: &Path, tok: &Tok, cfg: &S2Config) -> Result<SeqData> {
    let data = data::load_cfg(&[path.to_path_buf()], tok, cfg)?;
    if data.skipped > 0 {
        bail!("{} invalid sequence records", data.skipped);
    }
    // The loader keeps the file order within each split: rebuild the train sequences.
    let mut frames: Vec<(u64, usize)> = Vec::new();
    for file in list_jsonl(path)? {
        let f = std::fs::File::open(&file)?;
        for line in BufReader::new(f).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let v: serde_json::Value = serde_json::from_str(&line)?;
            let split = v["split"].as_str().unwrap_or("train");
            if split != "val" && split != "test" {
                frames.push((
                    v["seq"].as_u64().context("record without seq")?,
                    v["seq_t"].as_u64().context("record without seq_t")? as usize,
                ));
            }
        }
    }
    if frames.len() != data.train[0].len() {
        bail!(
            "{} train frames but {} train items",
            frames.len(),
            data.train[0].len()
        );
    }
    let mut by_seq: HashMap<u64, Vec<(usize, usize)>> = HashMap::new();
    for (i, (s, t)) in frames.iter().enumerate() {
        by_seq.entry(*s).or_default().push((*t, i));
    }
    let k = by_seq.values().map(Vec::len).max().unwrap_or(0);
    let mut train: Vec<Vec<usize>> = Vec::with_capacity(by_seq.len());
    for (id, mut v) in by_seq {
        v.sort_unstable();
        if v.iter().enumerate().any(|(t, (ft, _))| *ft != t) {
            bail!(
                "sequence {id}: frames {:?} are not 0..{}",
                v.iter().map(|x| x.0).collect::<Vec<_>>(),
                v.len()
            );
        }
        if v.len() != k {
            bail!("sequence {id} has {} frames, others {k}", v.len());
        }
        let idx: Vec<usize> = v.into_iter().map(|(_, i)| i).collect();
        if idx
            .windows(2)
            .any(|w| data.train[0][w[0]].stat != data.train[0][w[1]].stat)
        {
            bail!("sequence {id} mixes questions");
        }
        train.push(idx);
    }
    train.sort_unstable();
    Ok(SeqData { data, train, k })
}

/// One temporal training batch: `K` frame batches over the same rows (one
/// shared padded state length), the carry mask `[K, B]` and the carry source
/// exits (one-hot `[K, E]`).
pub struct SeqBatch {
    pub frames: Vec<S2Batch>,
    pub on: mlx_rs::Array,
    pub src: mlx_rs::Array,
    pub n_rows: usize,
    pub n_tokens: usize,
}

/// Samples `n` sequences; a frame's carry is on when it is not the first,
/// its state ids equal the previous frame's and it survives `drop` (p = 0.3).
pub fn sample(
    sd: &SeqData,
    n_exits: usize,
    n: usize,
    drop: f64,
    rng: &mut Rng,
    pad_to: usize,
) -> SeqBatch {
    let seqs: Vec<&Vec<usize>> = (0..n)
        .map(|_| &sd.train[rng.below(sd.train.len())])
        .collect();
    let items = &sd.data.train[0];
    let s_min = seqs
        .iter()
        .flat_map(|s| s.iter().map(|&i| items[i].state.len()))
        .max()
        .unwrap_or(1);
    let k = sd.k;
    let mut on = vec![0f32; k * n];
    let mut frames = Vec::with_capacity(k);
    let mut n_tokens = 0;
    for t in 0..k {
        let rows: Vec<&data::S2Item> = seqs.iter().map(|s| &items[s[t]]).collect();
        if t > 0 {
            for (r, s) in seqs.iter().enumerate() {
                let same = items[s[t]].state.ids == items[s[t - 1]].state.ids;
                on[t * n + r] = (same && rng.f64() >= drop) as u8 as f32;
            }
        }
        let b = make_batch_min(&sd.data.statics, &rows, pad_to, s_min);
        n_tokens += b.n_tokens;
        frames.push(b);
    }
    let mut src = vec![0f32; k * n_exits];
    for t in 0..k {
        src[t * n_exits + rng.below(n_exits)] = 1.0;
    }
    SeqBatch {
        frames,
        on: mlx_rs::Array::from_slice(&on, &[k as i32, n as i32]),
        src: mlx_rs::Array::from_slice(&src, &[k as i32, n_exits as i32]),
        n_rows: n * k,
        n_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_ids_differ_across_games() {
        // "flappy" and "rescue" have the same length: the ids must still differ.
        assert_ne!(seq_id("flappy", 11, 5), seq_id("rescue", 11, 5));
        assert_ne!(seq_id("dodge", 11, 5), seq_id("dodge", 12, 5));
        assert_ne!(seq_id("dodge", 11, 5), seq_id("dodge", 11, 6));
    }

    #[test]
    fn sequences_are_consecutive_frames_of_one_question() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("seq/flappy.jsonl");
        generate("flappy", 40, 4, 3, &out).unwrap();
        let lines: Vec<serde_json::Value> = std::fs::read_to_string(&out)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines.len(), 160);
        for c in lines.chunks(4) {
            let id = &c[0]["seq"];
            for (t, r) in c.iter().enumerate() {
                assert_eq!(&r["seq"], id);
                assert_eq!(r["seq_t"], t);
                assert_eq!(r["ins"], c[0]["ins"]);
                assert_eq!(r["opts"], c[0]["opts"]);
                assert_eq!(r["split"], c[0]["split"]);
                let tg: Vec<f64> = r["target"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap())
                    .collect();
                assert!((tg.iter().sum::<f64>() - 1.0).abs() < 1e-3);
            }
            // Consecutive decision points: the states differ but keep their keys.
            let keys = |s: &serde_json::Value| -> Vec<String> {
                s.as_str()
                    .unwrap()
                    .split(' ')
                    .step_by(2)
                    .map(String::from)
                    .collect()
            };
            assert_eq!(keys(&c[0]["state"]), keys(&c[3]["state"]));
        }
        // Loading rebuilds the train sequences in frame order.
        let _g = crate::test_util::lock();
        let (_d, tok) = crate::s2::data::tests::s2_tok();
        let mut cfg = S2Config::tiny(262);
        cfg.fuse_numbers = true;
        cfg.max_state_tokens = 64;
        let sd = load(&dir.path().join("seq"), &tok, &cfg).unwrap();
        assert_eq!(sd.k, 4);
        assert!(!sd.train.is_empty());
        let items = &sd.data.train[0];
        for s in &sd.train {
            assert_eq!(s.len(), 4);
            assert!(s.windows(2).all(|w| items[w[0]].stat == items[w[1]].stat));
        }
        let mut rng = Rng::new(1);
        let b = sample(&sd, 2, 8, 0.3, &mut rng, 16);
        assert_eq!(b.frames.len(), 4);
        let s0 = b.frames[0].ids.shape().to_vec();
        assert!(b.frames.iter().all(|f| f.ids.shape() == s0.as_slice()));
        b.on.eval().unwrap();
        let on = b.on.as_slice::<f32>();
        assert!(on[..8].iter().all(|&x| x == 0.0));
        // With fused numbers, flappy frames share their ids: most carries are on.
        assert!(on[8..].iter().sum::<f32>() >= 10.0, "{on:?}");
    }
}
