//! Calibration for the Q8_0 path.
//!
//! The residual stream of these small encoders has a few "massive" channels (~20x the others).
//! Inside a Q8_0 block such a channel sets the scale of the 31 others, and its weight rounding
//! error is amplified 20x. As in LLM.int8() (Dettmers et al., 2022), the input channels whose
//! calibrated magnitude stands far above the rest are computed in f32 with exact weight columns,
//! while everything else stays an int8 x int8 product. [`Calibration`] collects the per-channel
//! input magnitudes of every matrix product with a forward pass over [`CALIBRATION_PROMPTS`].

use crate::kernels::{BlockQ8, QK};

/// Input statistics of one matrix product ("site").
#[derive(Clone, Debug)]
pub struct SiteStats {
    pub k: usize,
    pub n_rows: usize,
    /// Largest `|x|` per input channel.
    pub absmax: Vec<f32>,
}

/// Calibration statistics of every matrix product of a model, indexed by site.
#[derive(Clone, Debug, Default)]
pub struct Calibration {
    pub sites: Vec<Option<SiteStats>>,
}

impl Calibration {
    pub fn new(n_sites: usize) -> Self {
        Calibration {
            sites: vec![None; n_sites],
        }
    }

    /// Accumulate `rows` input rows of width `k` for `site`.
    pub fn record(&mut self, site: usize, x: &[f32], rows: usize, k: usize) {
        if site >= self.sites.len() {
            self.sites.resize(site + 1, None);
        }
        let s = self.sites[site].get_or_insert_with(|| SiteStats {
            k,
            n_rows: 0,
            absmax: vec![0.0; k],
        });
        if s.k != k {
            return;
        }
        for row in x[..rows * k].chunks_exact(k) {
            for (m, &v) in s.absmax.iter_mut().zip(row) {
                *m = m.max(v.abs());
            }
        }
        s.n_rows += rows;
    }
}

/// Default outlier threshold: a channel is an outlier when its calibrated `max |x|` exceeds
/// this multiple of the median channel (at most `k / 16` channels per matrix).
pub const OUTLIER_RATIO: f32 = 3.0;

/// Outlier channels of a site: `max |x| > ratio * median`, largest first, at most `k / 16`.
pub fn outlier_channels(stats: &SiteStats, ratio: f32) -> Vec<usize> {
    let k = stats.k;
    let mut sorted = stats.absmax.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let median = sorted[k / 2];
    if median <= 0.0 {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..k)
        .filter(|&c| stats.absmax[c] > ratio * median)
        .collect();
    idx.sort_by(|&a, &b| stats.absmax[b].total_cmp(&stats.absmax[a]));
    idx.truncate((k / 16).max(1));
    idx.sort_unstable();
    idx
}

/// Quantize `w: [n][k]` to standard Q8_0 blocks, choosing in each block the scale that keeps the
/// weights of the input channels `cols` (the outlier channels) nearly exact. The runtime computes
/// those channels in f32 from the stored values, and their error is amplified by the large
/// activations, so they weigh `COL_WEIGHT` times more than the others. Same bytes as Q8_0.
pub fn quantize_q8_exact_cols(w: &[f32], n: usize, k: usize, cols: &[usize]) -> Vec<BlockQ8> {
    const COL_WEIGHT: f32 = 1e4;
    let nb = k / QK;
    let mut out = vec![BlockQ8::default(); n * nb];
    crate::kernels::quantize_q8(w, &mut out);
    let f16r = |d: f32| sky_gguf::f16_to_f32(sky_gguf::f32_to_f16(d));
    for b in 0..nb {
        let in_block: Vec<usize> = cols
            .iter()
            .filter(|&&c| c / QK == b)
            .map(|&c| c % QK)
            .collect();
        if in_block.is_empty() {
            continue;
        }
        for o in 0..n {
            let blk = &w[o * k + b * QK..o * k + (b + 1) * QK];
            let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            if amax == 0.0 {
                continue;
            }
            let err = |d: f32| -> f32 {
                let id = 1.0 / d;
                let mut e = 0.0;
                for (i, &v) in blk.iter().enumerate() {
                    let q = (v * id).round().clamp(-128.0, 127.0);
                    let r = (v - q * d) * (v - q * d);
                    e += if in_block.contains(&i) {
                        COL_WEIGHT * r
                    } else {
                        r
                    };
                }
                e
            };
            let d0 = f16r(amax / 127.0);
            let mut best = (err(d0), d0);
            // scales that make one outlier weight an exact multiple, without clipping the block
            for &i in &in_block {
                let v = blk[i].abs();
                if v == 0.0 {
                    continue;
                }
                let qmax = (127.0 * v / amax).floor().max(1.0) as i32;
                for q in (qmax - 24).max(1)..=qmax {
                    let d = f16r(v / q as f32);
                    if d * 127.5 < amax {
                        continue;
                    }
                    let e = err(d);
                    if e < best.0 {
                        best = (e, d);
                    }
                }
            }
            let d = best.1;
            let dst = &mut out[o * nb + b];
            dst.d = d;
            for (q, &v) in dst.qs.iter_mut().zip(blk) {
                *q = (v / d).round().clamp(-128.0, 127.0) as i8;
            }
        }
    }
    out
}

/// Built-in calibration prompts `(type, instructions, rendered options, state)`: generic
/// decision questions over game states, prose (English / French) and JSON.
pub const CALIBRATION_PROMPTS: &[(&str, &str, &[&str], &str)] = &[
    ("choice", "Which way should the player move to reach the exit?", &["left", "right", "up", "down"],
     "player x 3 y 5 exit x 9 y 1 wall east 2 lives 3"),
    ("choice", "What should the pilot do now?", &["climb: gain altitude", "hold: keep the course", "descend: lose altitude"],
     "alt 18 vy 4 dx 22 gt 14 gb 29 wind -2"),
    ("score", "How urgent is the request?", &["level 0: not urgent", "level 1: can wait a day", "level 2: today", "level 3: right now"],
     "Hi team, the build server has been down since this morning and nobody can deploy."),
    ("noul", "The text describes a sunny day.", &["false: no, the statement does not hold", "true: yes, the statement holds"],
     "Dark clouds rolled in over the hills and the rain did not stop until the evening."),
    ("choice", "Quel est le sujet principal du message ?", &["sport", "politique", "cuisine", "météo", "musique"],
     "Hier soir, l'équipe a gagné le match 3 à 1 grâce à deux buts en fin de partie. Les supporters étaient ravis."),
    ("noul", "Is the user asking for a refund?", &["false: no", "true: yes"],
     "{\"order\": 1832, \"status\": \"delivered\", \"message\": \"The item arrived broken, I want my money back.\"}"),
    ("score", "How full is the battery?", &["level 0: empty", "level 1: low", "level 2: half", "level 3: high", "level 4: full"],
     "battery 61 temp 34 load 12 fan 2"),
    ("choice", "Which item should the robot pick first?", &["box: the red box near the door", "ball: the small ball under the table",
     "cup: the cup on the shelf", "book: the open book", "key: the key on the floor", "bag: the bag by the window", "none"],
     "robot at 2 4 holding nothing; the door is open; a cat sleeps on the sofa; the lights are off in the kitchen."),
    ("noul", "Le texte est écrit à la première personne.", &["false: non", "true: oui"],
     "Je suis arrivé en retard au travail parce que le train était bloqué à cause de la neige."),
    ("choice", "Pick the best reply.", &["a: thanks, that helps a lot", "b: I do not understand the question", "c: please send the file again"],
     "User: I attached the report but I am not sure the numbers in table 2 are right. Can you check them?"),
    ("score", "Rate the difficulty of the level.", &["level 0: trivial", "level 1: easy", "level 2: medium", "level 3: hard", "level 4: very hard", "level 5: impossible"],
     "enemies 14 speed 7 gaps 5 time 30 checkpoints 0 ammo 3"),
    ("choice", "Which number is the largest?", &["-12", "7", "1003", "42", "0"], "values: -12, 7, 1003, 42, 0 (units: meters)"),
];

/// Token sequences `(ids, markers, qtype)` of [`CALIBRATION_PROMPTS`].
pub fn calibration_sequences(
    tok: &crate::Tokenizer,
    cfg: &crate::ModelConfig,
) -> crate::Result<Vec<crate::model::CalibSeq>> {
    let mut out = Vec::new();
    for (t, ins, opts, state) in CALIBRATION_PROMPTS {
        let q = crate::QType::from_name(t).expect("valid type");
        let (ids, markers) =
            crate::prompt::build_sequence(tok, q, ins, opts, state, cfg.max_len, cfg.head_max_len)?;
        if markers.len() == opts.len() {
            out.push((ids, markers, q.index()));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_columns_are_nearly_exact() {
        let (n, k) = (8, 64);
        let w: Vec<f32> = (0..n * k)
            .map(|i| (((i * 7919) % 1000) as f32 / 500.0 - 1.0) * 0.1)
            .collect();
        let q = quantize_q8_exact_cols(&w, n, k, &[3, 40]);
        let mut d = vec![0.0; n * k];
        crate::kernels::dequantize_q8(&q, &mut d);
        let mut plain = vec![BlockQ8::default(); n * k / QK];
        crate::kernels::quantize_q8(&w, &mut plain);
        let mut dp = vec![0.0; n * k];
        crate::kernels::dequantize_q8(&plain, &mut dp);
        let col_err = |v: &[f32], c: usize| {
            (0..n)
                .map(|o| (v[o * k + c] - w[o * k + c]).abs())
                .fold(0.0, f32::max)
        };
        for c in [3, 40] {
            assert!(
                col_err(&d, c) < 0.2 * col_err(&dp, c) + 1e-6,
                "column {c}: {} vs {}",
                col_err(&d, c),
                col_err(&dp, c)
            );
        }
        // the other columns stay about as accurate
        let rest = |v: &[f32]| {
            (0..n * k)
                .filter(|i| i % k != 3 && i % k != 40)
                .map(|i| (v[i] - w[i]).powi(2))
                .sum::<f32>()
        };
        assert!(rest(&d) < 2.0 * rest(&dp));
    }

    #[test]
    fn finds_massive_channels() {
        let k = 64;
        let mut cal = Calibration::new(1);
        let mut x = vec![0.0f32; 3 * k];
        for r in 0..3 {
            for c in 0..k {
                x[r * k + c] = ((r * 31 + c * 7) % 11) as f32 / 10.0 - 0.5;
            }
            x[r * k + 5] = 20.0;
            x[r * k + 40] = -15.0;
        }
        cal.record(0, &x, 3, k);
        let st = cal.sites[0].as_ref().unwrap();
        assert_eq!(st.n_rows, 3);
        assert_eq!(outlier_channels(st, 6.0), vec![5, 40]);
    }
}
