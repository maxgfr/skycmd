//! `sky-train s2-numeric`: the programmatic numeric curriculum of SPEC-S3 §3.
//!
//! Every record holds a `key value` state such as `a 12 b -3 c 40` and a
//! question whose gold answer is computed exactly from the values: largest /
//! smallest key (choice), `a > b` (noul), closest to a target (choice), sign of
//! `a − b` (choice), a 5-level bucket of one value (score) and `a` between `b`
//! and `c` (noul), each with English and French instructions. These records
//! teach the value channel order and magnitude; they are mixed into
//! `s2-pretrain` (as MLM text) and `s2-decide` (as decisions).

use std::{io::Write, path::Path};

use anyhow::Result;

use crate::{rng::Rng, sequence::DecisionRecord};

/// Question families, in generation order.
pub const FAMILIES: [&str; 6] = ["extreme", "greater", "closest", "sign", "bucket", "between"];

const KEYS: [&str; 32] = [
    "a", "b", "c", "d", "e", "x", "y", "z", "vy", "dx", "up", "dn", "alt", "bat", "wind", "hp",
    "am", "spd", "dist", "gap", "eta", "temp", "v", "dz", "hr", "gw", "cl", "nh", "sb", "lvl",
    "fuel", "load",
];

const NO_EN: &str = "false: no, the statement does not hold";
const YES_EN: &str = "true: yes, the statement holds";
const NO_FR: &str = "false: non, l'affirmation est fausse";
const YES_FR: &str = "true: oui, l'affirmation est vraie";

/// A value range of one record (all its values come from the same range so
/// comparisons are not trivial).
#[derive(Clone, Copy)]
enum Scale {
    Small,
    Mid,
    Positive,
    Decimal,
}

fn scale(rng: &mut Rng) -> Scale {
    match rng.below(4) {
        0 => Scale::Small,
        1 => Scale::Mid,
        2 => Scale::Positive,
        _ => Scale::Decimal,
    }
}

fn draw(rng: &mut Rng, s: Scale) -> f64 {
    match s {
        Scale::Small => rng.below(21) as f64 - 10.0,
        Scale::Mid => rng.below(201) as f64 - 100.0,
        Scale::Positive => rng.below(1001) as f64,
        Scale::Decimal => (rng.below(401) as f64 - 200.0) / 10.0,
    }
}

/// Number text as it appears in game states: integers without a decimal
/// point, otherwise one decimal.
pub fn fmt(v: f64) -> String {
    if v.fract() == 0.0 {
        format!("{}", v as i64)
    } else {
        format!("{v:.1}")
    }
}

/// `k` distinct keys in random order.
fn keys(rng: &mut Rng, k: usize) -> Vec<&'static str> {
    let mut all: Vec<&'static str> = KEYS.to_vec();
    rng.shuffle(&mut all);
    all.truncate(k);
    all
}

/// `k` distinct values of one scale.
fn values(rng: &mut Rng, s: Scale, k: usize) -> Vec<f64> {
    let mut out: Vec<f64> = Vec::with_capacity(k);
    while out.len() < k {
        let v = draw(rng, s);
        if !out.contains(&v) {
            out.push(v);
        }
    }
    out
}

fn state_text(ks: &[&str], vs: &[f64]) -> String {
    ks.iter()
        .zip(vs)
        .map(|(k, v)| format!("{k} {}", fmt(*v)))
        .collect::<Vec<_>>()
        .join(" ")
}

fn one_hot(k: usize, i: usize) -> Vec<f32> {
    (0..k).map(|j| if j == i { 1.0 } else { 0.0 }).collect()
}

fn noul(fr: bool, truth: bool) -> (Vec<String>, Vec<f32>) {
    let opts = if fr { [NO_FR, YES_FR] } else { [NO_EN, YES_EN] };
    (
        opts.iter().map(|s| s.to_string()).collect(),
        one_hot(2, truth as usize),
    )
}

/// Options = the keys in a random order; the gold one is `gold`.
fn key_options(rng: &mut Rng, ks: &[&str], gold: usize) -> (Vec<String>, Vec<f32>) {
    let mut order: Vec<usize> = (0..ks.len()).collect();
    rng.shuffle(&mut order);
    let opts = order.iter().map(|&i| ks[i].to_string()).collect();
    let at = order.iter().position(|&i| i == gold).unwrap_or(0);
    (opts, one_hot(ks.len(), at))
}

fn pick<'a>(rng: &mut Rng, xs: &[&'a str]) -> &'a str {
    xs[rng.below(xs.len())]
}

/// One record of family `fam` (index into [`FAMILIES`]).
pub fn record(rng: &mut Rng, fam: usize) -> DecisionRecord {
    let fr = rng.below(2) == 1;
    let s = scale(rng);
    let (t, ins, state, opts, target): (&str, String, String, Vec<String>, Vec<f32>) = match fam {
        0 => {
            let k = 3 + rng.below(4);
            let (ks, vs) = (keys(rng, k), values(rng, s, k));
            let largest = rng.below(2) == 0;
            let gold = (0..k).fold(0, |b, i| {
                if (vs[i] > vs[b]) == largest && vs[i] != vs[b] {
                    i
                } else {
                    b
                }
            });
            let ins = match (fr, largest) {
                (false, true) => pick(
                    rng,
                    &[
                        "Which key has the largest value?",
                        "Which reading is the highest?",
                        "Pick the key with the biggest number.",
                    ],
                ),
                (false, false) => pick(
                    rng,
                    &[
                        "Which key has the smallest value?",
                        "Which reading is the lowest?",
                        "Pick the key with the smallest number.",
                    ],
                ),
                (true, true) => pick(
                    rng,
                    &[
                        "Quelle clé a la plus grande valeur ?",
                        "Quelle mesure est la plus élevée ?",
                    ],
                ),
                (true, false) => pick(
                    rng,
                    &[
                        "Quelle clé a la plus petite valeur ?",
                        "Quelle mesure est la plus basse ?",
                    ],
                ),
            }
            .to_string();
            let (opts, target) = key_options(rng, &ks, gold);
            ("choice", ins, state_text(&ks, &vs), opts, target)
        }
        1 => {
            let k = 2 + rng.below(5);
            let (ks, mut vs) = (keys(rng, k), values(rng, s, k));
            let (i, j) = (0, 1 + rng.below(k - 1));
            if rng.below(8) == 0 {
                vs[j] = vs[i]; // equality: "greater" is false
            }
            let less = rng.below(2) == 0;
            let truth = if less { vs[i] < vs[j] } else { vs[i] > vs[j] };
            let (a, b) = (ks[i], ks[j]);
            let ins = match (fr, less) {
                (false, false) => format!("Is {a} greater than {b}?"),
                (false, true) => format!("Is {a} less than {b}?"),
                (true, false) => format!("{a} est-il plus grand que {b} ?"),
                (true, true) => format!("{a} est-il plus petit que {b} ?"),
            };
            let mut order: Vec<usize> = (0..k).collect();
            rng.shuffle(&mut order);
            let sk: Vec<&str> = order.iter().map(|&o| ks[o]).collect();
            let sv: Vec<f64> = order.iter().map(|&o| vs[o]).collect();
            let (opts, target) = noul(fr, truth);
            ("noul", ins, state_text(&sk, &sv), opts, target)
        }
        2 => loop {
            let k = 3 + rng.below(4);
            let (ks, vs) = (keys(rng, k), values(rng, s, k));
            let tgt = draw(rng, s);
            let dist: Vec<f64> = vs.iter().map(|v| (v - tgt).abs()).collect();
            let gold = (0..k).fold(0, |b, i| if dist[i] < dist[b] { i } else { b });
            if dist.iter().filter(|&&d| d == dist[gold]).count() > 1 {
                continue;
            }
            // The target lives in the question half of the time, in the state otherwise.
            let in_state = rng.below(2) == 0;
            let (ins, state) = if in_state {
                let tk = if fr { "cible" } else { "target" };
                let ins = if fr {
                    format!("Quelle clé est la plus proche de {tk} ?")
                } else {
                    format!("Which key is closest to {tk}?")
                };
                (ins, format!("{tk} {} {}", fmt(tgt), state_text(&ks, &vs)))
            } else {
                let ins = if fr {
                    format!("Quelle valeur est la plus proche de {} ?", fmt(tgt))
                } else {
                    format!("Which value is closest to {}?", fmt(tgt))
                };
                (ins, state_text(&ks, &vs))
            };
            let (opts, target) = key_options(rng, &ks, gold);
            break ("choice", ins, state, opts, target);
        },
        3 => {
            let k = 2 + rng.below(4);
            let (ks, mut vs) = (keys(rng, k), values(rng, s, k));
            if rng.below(6) == 0 {
                vs[1] = vs[0];
            }
            let (a, b) = (ks[0], ks[1]);
            let d = vs[0] - vs[1];
            let gold = if d > 0.0 {
                0
            } else if d == 0.0 {
                1
            } else {
                2
            };
            let (ins, names) = if fr {
                (
                    format!("Quel est le signe de {a} moins {b} ?"),
                    ["positif", "nul", "négatif"],
                )
            } else {
                (
                    format!("What is the sign of {a} minus {b}?"),
                    ["positive", "zero", "negative"],
                )
            };
            let mut order: Vec<usize> = (0..k).collect();
            rng.shuffle(&mut order);
            let sk: Vec<&str> = order.iter().map(|&o| ks[o]).collect();
            let sv: Vec<f64> = order.iter().map(|&o| vs[o]).collect();
            (
                "choice",
                ins,
                state_text(&sk, &sv),
                names.iter().map(|x| x.to_string()).collect(),
                one_hot(3, gold),
            )
        }
        4 => loop {
            let k = 1 + rng.below(4);
            let (ks, vs) = (keys(rng, k), values(rng, s, k));
            let lo = draw(rng, s);
            let step = match s {
                Scale::Small => 1.0 + rng.below(3) as f64,
                Scale::Mid => 5.0 * (1 + rng.below(6)) as f64,
                Scale::Positive => 25.0 * (1 + rng.below(8)) as f64,
                Scale::Decimal => 0.5 * (1 + rng.below(8)) as f64,
            };
            let th: Vec<f64> = (0..4)
                .map(|i| ((lo + step * i as f64) * 10.0).round() / 10.0)
                .collect();
            let i = rng.below(k);
            let v = vs[i];
            if th.contains(&v) {
                continue;
            }
            let level = th.iter().filter(|&&x| x < v).count();
            let opts: Vec<String> = (0..5)
                .map(|l| {
                    let desc = match (l, fr) {
                        (0, false) => format!("below {}", fmt(th[0])),
                        (4, false) => format!("above {}", fmt(th[3])),
                        (_, false) => format!("{} to {}", fmt(th[l - 1]), fmt(th[l])),
                        (0, true) => format!("moins de {}", fmt(th[0])),
                        (4, true) => format!("plus de {}", fmt(th[3])),
                        (_, true) => format!("de {} à {}", fmt(th[l - 1]), fmt(th[l])),
                    };
                    format!("level {l}: {desc}")
                })
                .collect();
            let ins = if fr {
                format!("Classe la valeur de {} dans un des 5 niveaux.", ks[i])
            } else {
                format!("Bucket the value of {} into one of 5 levels.", ks[i])
            };
            break ("score", ins, state_text(&ks, &vs), opts, one_hot(5, level));
        },
        _ => {
            let k = 3 + rng.below(4);
            let (ks, mut vs) = (keys(rng, k), values(rng, s, k));
            // Half of the time force a to lie inside the interval.
            if rng.below(2) == 0 {
                let (lo, hi) = (vs[1].min(vs[2]), vs[1].max(vs[2]));
                if hi - lo > 1.0 {
                    let mid = ((lo + hi) / 2.0).round();
                    if mid > lo && mid < hi && !vs.contains(&mid) {
                        vs[0] = mid;
                    }
                }
            }
            let (lo, hi) = (vs[1].min(vs[2]), vs[1].max(vs[2]));
            let truth = vs[0] > lo && vs[0] < hi;
            let (a, b, c) = (ks[0], ks[1], ks[2]);
            let ins = if fr {
                format!("{a} est-il compris entre {b} et {c} ?")
            } else {
                format!("Does {a} lie between {b} and {c}?")
            };
            let mut order: Vec<usize> = (0..k).collect();
            rng.shuffle(&mut order);
            let sk: Vec<&str> = order.iter().map(|&o| ks[o]).collect();
            let sv: Vec<f64> = order.iter().map(|&o| vs[o]).collect();
            let (opts, target) = noul(fr, truth);
            ("noul", ins, state_text(&sk, &sv), opts, target)
        }
    };
    let split = match rng.below(20) {
        0 => "val",
        1 => "test",
        _ => "train",
    };
    DecisionRecord {
        src: format!("numeric:{}", FAMILIES[fam.min(FAMILIES.len() - 1)]),
        t: t.into(),
        ins,
        opts,
        state,
        target,
        split: split.into(),
    }
}

/// `n` records, families in turn.
pub fn generate(n: usize, seed: u64) -> Vec<DecisionRecord> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|i| record(&mut rng, i % FAMILIES.len()))
        .collect()
}

/// Writes `n` records as JSONL.
pub fn write(n: usize, seed: u64, out: &Path) -> Result<()> {
    if let Some(d) = out.parent() {
        std::fs::create_dir_all(d)?;
    }
    let mut w = std::io::BufWriter::new(std::fs::File::create(out)?);
    for r in generate(n, seed) {
        serde_json::to_writer(&mut w, &r)?;
        w.write_all(b"\n")?;
    }
    w.flush()?;
    Ok(())
}

/// MLM text of a numeric record: the state, the question and the gold answer.
pub fn mlm_text(r: &DecisionRecord) -> String {
    let gold = (0..r.target.len()).fold(0, |b, i| if r.target[i] > r.target[b] { i } else { b });
    format!(
        "{}. {} {}",
        r.state,
        r.ins,
        r.opts.get(gold).cloned().unwrap_or_default()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::num::extract;

    fn parse(state: &str) -> Vec<(String, f64)> {
        let w: Vec<&str> = state.split(' ').collect();
        w.chunks(2)
            .map(|c| (c[0].to_string(), c[1].parse().unwrap()))
            .collect()
    }

    #[test]
    fn records_have_exact_gold() {
        let recs = generate(6000, 3);
        let mut fr = 0;
        let mut splits = [0usize; 3];
        let mut yes = 0;
        let mut nouls = 0;
        for r in &recs {
            assert!(!r.opts.is_empty() && r.target.len() == r.opts.len());
            assert!((r.target.iter().sum::<f32>() - 1.0).abs() < 1e-6);
            let gold = r.target.iter().position(|&x| x == 1.0).unwrap();
            fr += r.ins.contains('?') as usize * r.ins.contains(" ?") as usize;
            splits[["train", "val", "test"]
                .iter()
                .position(|s| *s == r.split)
                .unwrap()] += 1;
            // Every number of the state is extracted (one [NUM] per value).
            let kv = parse(&r.state);
            assert_eq!(
                extract(&r.state).values.len(),
                r.state.split(' ').count() / 2
            );
            let val = |k: &str| kv.iter().find(|(n, _)| n == k).map(|x| x.1);
            match r.src.as_str() {
                "numeric:extreme" => {
                    let v: Vec<f64> = r.opts.iter().map(|o| val(o).unwrap()).collect();
                    let largest = r.ins.contains("larg")
                        || r.ins.contains("high")
                        || r.ins.contains("big")
                        || r.ins.contains("grande")
                        || r.ins.contains("élevée");
                    let best = v.iter().cloned().fold(
                        if largest { f64::MIN } else { f64::MAX },
                        if largest { f64::max } else { f64::min },
                    );
                    assert_eq!(v[gold], best, "{r:?}");
                }
                "numeric:greater" | "numeric:between" => {
                    nouls += 1;
                    yes += gold;
                }
                "numeric:sign" => assert_eq!(r.opts.len(), 3),
                "numeric:bucket" => assert_eq!(r.opts.len(), 5),
                _ => {}
            }
        }
        assert!(fr > 1000, "French instructions: {fr}");
        assert!(splits[1] > 150 && splits[2] > 150);
        let rate = yes as f64 / nouls as f64;
        assert!((0.3..0.7).contains(&rate), "noul true rate {rate}");
        assert!(mlm_text(&recs[0]).starts_with(&recs[0].state));
    }
}
