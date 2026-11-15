//! `sky-train s2-memcheck`: the checks C1 to C3 of `docs/s6-sleep2-prereg.md` §3, before any
//! training, on the CPU and on train seeds only.
//!
//! - **C1**, the retrieval algebra ([`memory::retrieval_algebra`], also a unit test of
//!   `s2::memory`).
//! - **C2**, the colour read from the memory at initialisation: the base with fresh memory
//!   weights (as `s2-decide --init` builds them at step 0, QAT on) plays `recall` train episodes
//!   through the memory; `r = merge(S·q)`, before the read-out, is taken at the door token of each
//!   ask frame, at the last block application, and its colour is read by a nearest centroid on
//!   cosine (centroids from the first half of the episodes, accuracy on the other half). Also: the
//!   mean squared cosine between the door keys of the show frames and the keys of the idle frames
//!   of the same episode, and the colour read the same way from the base's own hidden state at the
//!   door token of the show frames.
//! - **C3**, the first gradient: `∂L/∂Wo` at step 0 on independent `recall` training batches, the
//!   mean of their pairwise cosines, the gradient norm of every parameter group, and the largest
//!   gradient of the other memory weights.
//!
//! Three memory forms are measured: v2 with `frame` values, v2 with `token` values, and v1
//! (separate `q`/`k`, `token` values) as the reference.

use std::{
    collections::{BTreeMap, HashMap},
    path::PathBuf,
    rc::Rc,
};

use anyhow::{bail, Context, Result};
use mlx_rs::{transforms::keyed_value_and_grad, Array};
use serde::Serialize;
use serde_json::json;
use sky_games::{gen::split_for_seed, make_game};

use crate::{
    model::host_f32,
    params::{self, ParamMap},
    rng::Rng,
    s2::{
        config::{MemoryQk, MemoryValue, MemoryWrite, S2Config},
        data::{encode_record_cfg, make_batch, S2Item, StaticSeq},
        memory::{self, Memory, Sleep, SLEEP_EVERY},
        model::{exit_weight_vec, expand_pretrained, init, prepare, state_written, S2Parts},
        seq::SleepData,
        train::group_norms,
    },
    sequence::{qtype_id, DecisionRecord, Tok},
};

/// C2's bar: the better v2 form reads the colour of the asks of delays 1–8 at least this often.
pub const C2_BAR: f64 = 0.40;
/// C2's tie rule: both forms over the bar and closer than this, `frame` is taken.
pub const C2_TIE: f64 = 0.02;
/// C3's bar: the mean pairwise cosine of the step-0 gradients of `Wo`.
pub const C3_BAR: f64 = 0.10;
/// The idle frames of C1 whose argmax must be red under form `frame`.
pub const C1_PASS_IDLE: [usize; 3] = [1, 4, 8];
/// The idle frames of C1 that are printed and recorded.
pub const C1_IDLE: [usize; 5] = [1, 4, 8, 16, 32];
const COLOURS: [&str; 4] = ["red", "green", "blue", "yellow"];
const DOORS: [&str; 3] = ["a", "b", "c"];

#[derive(clap::Args, Debug, Clone)]
pub struct MemcheckArgs {
    /// The base checkpoint (model.safetensors, config.json, tokenizer.json).
    #[arg(long)]
    pub base: PathBuf,
    /// `recall` training sequences (s2-seqgen output; its train split only is read) for C3.
    #[arg(long)]
    pub seq: PathBuf,
    /// Output directory: c1.json, c2.json, c3.json.
    #[arg(long)]
    pub out: PathBuf,
    /// C2: train episodes played (the first half gives the centroids).
    #[arg(long, default_value_t = 2000)]
    pub episodes: usize,
    /// C2: the episodes are the first train-split seeds from this one upward.
    #[arg(long, default_value_t = 0)]
    pub from_seed: u64,
    /// C2: episodes played side by side in one batch.
    #[arg(long, default_value_t = 50)]
    pub rows: usize,
    /// C3: independent batches.
    #[arg(long, default_value_t = 10)]
    pub batches: usize,
    /// C3: sequences per batch.
    #[arg(long, default_value_t = 16)]
    pub batch: usize,
    /// Seed of the fresh memory weights (as `s2-decide --seed`) and of C3's batches.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Padding multiple of C3's batches (the pilot's).
    #[arg(long, default_value_t = 32)]
    pub pad_to: usize,
    /// Run C1 only.
    #[arg(long)]
    pub c1_only: bool,
    /// The write rule of the two v2 forms (amendment 1: the one the sweep adopted); v1's
    /// reference keeps v1's rule.
    #[arg(long, default_value = "average", value_parser = parse_write)]
    pub write: MemoryWrite,
    /// The sweep of amendment 1 (write rules × passes × idle frames on C1's synthetic states,
    /// `sweep.json` and `sweep.md`) and its adoption rule; nothing else.
    #[arg(long)]
    pub sweep: bool,
    /// Only the gradient report of C3 on the three forms (`grads.json`), with no verdict: the
    /// step-0 gradient of every parameter group, v1 against v2.
    #[arg(long)]
    pub grads_only: bool,
}

fn parse_write(s: &str) -> std::result::Result<MemoryWrite, String> {
    MemoryWrite::parse(s).ok_or_else(|| format!("{s:?}: average, delta, exact or centred"))
}

/// One memory form of the checks.
#[derive(Debug, Clone, Copy)]
struct Form {
    name: &'static str,
    qk: MemoryQk,
    value: MemoryValue,
}

const FORMS: [Form; 3] = [
    Form {
        name: "frame",
        qk: MemoryQk::Shared,
        value: MemoryValue::Frame,
    },
    Form {
        name: "token",
        qk: MemoryQk::Shared,
        value: MemoryValue::Token,
    },
    Form {
        name: "v1",
        qk: MemoryQk::Separate,
        value: MemoryValue::Token,
    },
];

/// The configuration of `form` on the base's.
fn form_cfg(base: &S2Config, form: Form, write: MemoryWrite) -> Result<S2Config> {
    let mut c = base.clone();
    c.memory = true;
    c.memory_qk = form.qk;
    c.memory_value = form.value;
    if form.qk == MemoryQk::Shared {
        c.memory_write = write;
    }
    c.temperature.clear();
    c.validate()?;
    Ok(c)
}

/// The parameters of `cfg` at step 0 of `s2-decide --init <base> --seed <seed>`: a fresh
/// initialization, then every tensor of the base.
fn step0_params(cfg: &S2Config, base: &ParamMap, seed: u64) -> Result<ParamMap> {
    let mut p = init(cfg, S2Parts::DECISION, seed)?;
    expand_pretrained(&mut p, base, cfg)?;
    Ok(p)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f64, *y as f64);
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    if aa == 0.0 || bb == 0.0 {
        0.0
    } else {
        ab / (aa.sqrt() * bb.sqrt())
    }
}

/// Nearest centroid on cosine: centroids (mean of the unit vectors) from `train`, the share of
/// `test` read right, and the number of test vectors. `(vector, class)` pairs.
fn nearest_centroid(train: &[(Vec<f32>, usize)], test: &[(&[f32], usize)]) -> (f64, usize) {
    let dim = train.first().map_or(0, |(v, _)| v.len());
    let mut centroid = vec![vec![0f64; dim]; COLOURS.len()];
    for (v, c) in train {
        let n = v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        if n > 0.0 {
            for (s, x) in centroid[*c].iter_mut().zip(v) {
                *s += *x as f64 / n;
            }
        }
    }
    let centroid: Vec<Vec<f32>> = centroid
        .into_iter()
        .map(|c| c.into_iter().map(|x| x as f32).collect())
        .collect();
    let right = test
        .iter()
        .filter(|(v, c)| {
            let got = (0..COLOURS.len()).fold(0, |b, k| {
                if cosine(v, &centroid[k]) > cosine(v, &centroid[b]) {
                    k
                } else {
                    b
                }
            });
            got == *c
        })
        .count();
    (right as f64 / test.len().max(1) as f64, test.len())
}

/// What a frame of `recall` shows: `(shown door, shown colour, asked door)`.
fn parse(state: &str) -> Result<(Option<usize>, Option<usize>, Option<usize>)> {
    let w: Vec<&str> = state.split_whitespace().collect();
    if w.len() != 8 || w[0] != "show" || w[2] != "colour" || w[4] != "ask" || w[6] != "noise" {
        bail!("not a recall frame: {state:?}");
    }
    let door = |x: &str| DOORS.iter().position(|d| *d == x);
    let colour = COLOURS.iter().position(|c| *c == w[3]);
    Ok((door(w[1]), colour, door(w[5])))
}

/// One episode of `recall`: the state text of each frame.
fn episode(seed: u64) -> Result<Vec<String>> {
    let mut g = make_game("recall", seed).context("recall")?;
    let mut out = Vec::new();
    while !g.done() {
        if g.decision_due() {
            out.push(g.state_text());
            g.apply(0, false);
        } else {
            g.tick();
        }
    }
    Ok(out)
}

/// The first `n` train-split episode seeds from `from` upward.
pub fn train_seeds(from: u64, n: usize) -> Vec<u64> {
    let train = |s: &u64| serde_json::to_value(split_for_seed(*s)).is_ok_and(|v| v == "train");
    (from..).filter(train).take(n).collect()
}

/// Encodes the frames of recall as the probe reader does (its question, canonical options).
struct Encoder<'a> {
    tok: &'a Tok,
    cfg: &'a S2Config,
    rec: DecisionRecord,
    statics: Vec<StaticSeq>,
    qtype: usize,
    k: usize,
    /// Door position by the state text up to its noise number.
    door_at: HashMap<String, usize>,
}

impl<'a> Encoder<'a> {
    fn new(tok: &'a Tok, cfg: &'a S2Config) -> Result<Self> {
        let g = make_game("recall", 0).context("recall")?;
        let q = &g.questions()[0];
        let t = serde_json::to_value(q.qtype)?
            .as_str()
            .unwrap_or("choice")
            .to_string();
        let opts = q.canonical_options();
        let rec = DecisionRecord {
            src: "memcheck".into(),
            t: t.clone(),
            ins: q.instructions().to_string(),
            opts: opts.clone(),
            state: String::new(),
            target: vec![1.0; opts.len()],
            split: "train".into(),
        };
        let (st, _) = encode_record_cfg(tok, &rec, cfg)?;
        Ok(Encoder {
            tok,
            cfg,
            rec,
            statics: vec![st],
            qtype: qtype_id(&t).unwrap_or(0),
            k: opts.len(),
            door_at: HashMap::new(),
        })
    }

    fn item(&self, state: &str) -> Result<S2Item> {
        let rec = DecisionRecord {
            state: state.into(),
            ..self.rec.clone()
        };
        let (_, stream) = encode_record_cfg(self.tok, &rec, self.cfg)?;
        Ok(S2Item {
            stat: 0,
            state: stream,
            qtype: self.qtype,
            target: vec![1.0 / self.k as f32; self.k],
            src: "memcheck".into(),
            group: 0,
        })
    }

    /// The position of the door token of a show (`after = "show"`) or ask (`"ask"`) frame: the
    /// one token that changes when the door does.
    fn door(&mut self, state: &str, after: &str, door: usize) -> Result<usize> {
        let key = format!("{after}|{}", state.split(" noise").next().unwrap_or(state));
        if let Some(&at) = self.door_at.get(&key) {
            return Ok(at);
        }
        let other = DOORS[(door + 1) % DOORS.len()];
        let swapped = state.replacen(
            &format!("{after} {}", DOORS[door]),
            &format!("{after} {other}"),
            1,
        );
        let (a, b) = (self.item(state)?.state.ids, self.item(&swapped)?.state.ids);
        let diff: Vec<usize> = (0..a.len().min(b.len()))
            .filter(|&i| a[i] != b[i])
            .collect();
        if diff.len() != 1 || a.len() != b.len() {
            bail!("{state:?}: the door is not one token ({diff:?})");
        }
        self.door_at.insert(key, diff[0]);
        Ok(diff[0])
    }
}

/// Rows `[B, T, d]` of a host copy: the vector of row `r`, token `t`.
fn token_vec(flat: &[f32], t_len: usize, d: usize, r: usize, t: usize) -> &[f32] {
    &flat[(r * t_len + t) * d..(r * t_len + t + 1) * d]
}

#[derive(Debug, Clone, Serialize)]
pub struct C2Form {
    pub form: &'static str,
    pub memory_qk: &'static str,
    pub memory_value: &'static str,
    /// Nearest-centroid accuracy on the held-out asks of delays 1–8.
    pub accuracy_1_8: f64,
    pub asks_1_8: usize,
    /// On every held-out ask, and per delay bucket (1–4, 5–8, 9–16, 17–32).
    pub accuracy_all: f64,
    pub by_bucket: Vec<f64>,
    /// Mean squared cosine between the door keys of the show frames and the keys of the idle
    /// frames of the same episode (mean over heads).
    pub key_overlap: f64,
}

/// What C2 reads on one form, and the base's own door-token hidden states at the show frames.
struct Played {
    /// (episode, bucket, colour, r at the door token over the layers).
    asks: Vec<(usize, usize, usize, Vec<f32>)>,
    /// (episode, colour, the block output at the door token over the layers).
    shows: Vec<(usize, usize, Vec<f32>)>,
    key_overlap: f64,
}

fn play(
    p: &ParamMap,
    cfg: &S2Config,
    enc: &mut Encoder<'_>,
    seeds: &[u64],
    rows: usize,
) -> Result<Played> {
    let (h, hd, d) = (cfg.n_heads(), cfg.head_dim(), cfg.hidden_size);
    let mut out = Played {
        asks: Vec::new(),
        shows: Vec::new(),
        key_overlap: 0.0,
    };
    let (mut overlap, mut pairs) = (0f64, 0f64);
    for (chunk_i, chunk) in seeds.chunks(rows).enumerate() {
        let eps: Vec<Vec<String>> = chunk.iter().map(|&s| episode(s)).collect::<Result<_>>()?;
        let frames = eps[0].len();
        if eps.iter().any(|e| e.len() != frames) {
            bail!("recall episodes of different lengths");
        }
        let b = eps.len();
        let mut mem = Memory::zeros(cfg, b as i32)?;
        // Per row: the last show of each door (frame, colour), the door keys of its shows and
        // the keys of its idle frames, per head.
        let mut last_show = vec![[None::<(usize, usize)>; 3]; b];
        let mut door_keys: Vec<Vec<Vec<f32>>> = vec![Vec::new(); b];
        let mut idle_keys: Vec<Vec<Vec<f32>>> = vec![Vec::new(); b];
        for t in 0..frames {
            let items: Vec<S2Item> = eps.iter().map(|e| enc.item(&e[t])).collect::<Result<_>>()?;
            let refs: Vec<&S2Item> = items.iter().collect();
            let batch = make_batch(&enc.statics, &refs, 1);
            let written = state_written(p, cfg, &batch, &mem.0)?;
            let t_len = batch.ids.dim(1) as usize;
            let lens: Vec<usize> = items.iter().map(|i| i.state.len()).collect();
            let mut r_host = Vec::new();
            let mut x_host = Vec::new();
            let mut k_host = Vec::new();
            for (l, x) in written.iter().enumerate() {
                let x = x.as_ref().context("a block the frame did not run")?;
                r_host.push(host_f32(&memory::retrieve(p, cfg, l, x, &mem.0[l])?)?);
                x_host.push(host_f32(x)?);
                k_host.push(host_f32(&memory::keys(p, cfg, l, x)?)?);
            }
            let over_layers = |host: &[Vec<f32>], r: usize, at: usize| -> Vec<f32> {
                host.iter()
                    .flat_map(|f| token_vec(f, t_len, d, r, at).to_vec())
                    .collect()
            };
            // Keys `[B, H, T, hd]` of row r, token tt, every layer and head.
            let key_at = |r: usize, tt: usize| -> Vec<f32> {
                k_host
                    .iter()
                    .flat_map(|f| {
                        (0..h).flat_map(move |hh| {
                            let o = ((r * h + hh) * t_len + tt) * hd;
                            f[o..o + hd].to_vec()
                        })
                    })
                    .collect()
            };
            for (r, e) in eps.iter().enumerate() {
                let ep = chunk_i * rows + r;
                match parse(&e[t])? {
                    (Some(door), Some(colour), None) => {
                        let at = enc.door(&e[t], "show", door)?;
                        out.shows.push((ep, colour, over_layers(&x_host, r, at)));
                        door_keys[r].push(key_at(r, at));
                        last_show[r][door] = Some((t, colour));
                    }
                    (None, None, Some(door)) => {
                        let at = enc.door(&e[t], "ask", door)?;
                        let (shown, colour) = last_show[r][door]
                            .with_context(|| format!("ask of door {door} before its show"))?;
                        let bucket = sky_games::recall::bucket_of((t - shown) as u32);
                        out.asks
                            .push((ep, bucket, colour, over_layers(&r_host, r, at)));
                    }
                    (None, None, None) => {
                        for tt in 0..lens[r] {
                            idle_keys[r].push(key_at(r, tt));
                        }
                    }
                    other => bail!("unexpected recall frame {other:?}: {}", e[t]),
                }
            }
            mem = memory::after_writes(p, cfg, &written, &batch.valid, &mem)?;
            mlx_rs::transforms::eval(mem.0.iter())?;
        }
        // c̄²: per layer and head, the squared cosine of unit keys (one per head).
        let per = h * hd;
        for r in 0..b {
            for dk in &door_keys[r] {
                for ik in &idle_keys[r] {
                    for l in 0..dk.len() / per {
                        for hh in 0..h {
                            let o = l * per + hh * hd;
                            let dot: f32 = dk[o..o + hd]
                                .iter()
                                .zip(&ik[o..o + hd])
                                .map(|(x, y)| x * y)
                                .sum();
                            overlap += (dot as f64).powi(2);
                            pairs += 1.0;
                        }
                    }
                }
            }
        }
    }
    out.key_overlap = overlap / pairs.max(1.0);
    Ok(out)
}

/// The reading of C2 on one form.
fn read_form(form: Form, played: &Played, n_train: usize) -> C2Form {
    let train: Vec<(Vec<f32>, usize)> = played
        .asks
        .iter()
        .filter(|a| a.0 < n_train)
        .map(|a| (a.3.clone(), a.2))
        .collect();
    let test = |keep: &dyn Fn(usize) -> bool| -> (f64, usize) {
        let t: Vec<(&[f32], usize)> = played
            .asks
            .iter()
            .filter(|a| a.0 >= n_train && keep(a.1))
            .map(|a| (a.3.as_slice(), a.2))
            .collect();
        nearest_centroid(&train, &t)
    };
    let (acc, n) = test(&|b| b <= 1);
    C2Form {
        form: form.name,
        memory_qk: form.qk.name(),
        memory_value: form.value.name(),
        accuracy_1_8: acc,
        asks_1_8: n,
        accuracy_all: test(&|_| true).0,
        by_bucket: (0..4).map(|k| test(&|b| b == k).0).collect(),
        key_overlap: played.key_overlap,
    }
}

/// C2's rule: the adopted value form, if the better v2 form reaches the bar.
pub fn c2_adopt(frame: f64, token: f64) -> Option<&'static str> {
    let best = frame.max(token);
    if best < C2_BAR {
        None
    } else if (frame >= C2_BAR && token >= C2_BAR && (frame - token).abs() < C2_TIE)
        || frame >= token
    {
        // Both over the bar and closer than C2_TIE: `frame`; otherwise the better one.
        Some("frame")
    } else {
        Some("token")
    }
}

/// The mean of the pairwise cosines of `v`.
pub fn mean_pairwise_cosine(v: &[Vec<f32>]) -> f64 {
    let (mut s, mut n) = (0.0, 0usize);
    for i in 0..v.len() {
        for j in i + 1..v.len() {
            s += cosine(&v[i], &v[j]);
            n += 1;
        }
    }
    s / n.max(1) as f64
}

fn write_json(path: &std::path::Path, v: &serde_json::Value) -> Result<()> {
    std::fs::write(path, serde_json::to_string_pretty(v)?)
        .with_context(|| format!("writing {}", path.display()))
}

fn verdict(ok: bool) -> &'static str {
    if ok {
        "PASS"
    } else {
        "FAIL"
    }
}

fn c1(out: &std::path::Path) -> Result<bool> {
    let (cos, rows) = memory::retrieval_algebra(&C1_IDLE, 0)?;
    println!("C1 retrieval algebra (d = 96, mean cosine between words {cos:.3}):");
    for r in &rows {
        println!(
            "  form {:5} idle {:2}: red {:+.4} green {:+.4} blue {:+.4} yellow {:+.4} -> {}",
            r.form,
            r.idle,
            r.colours[0],
            r.colours[1],
            r.colours[2],
            r.colours[3],
            COLOURS[r.argmax]
        );
    }
    let pass = C1_PASS_IDLE.iter().all(|i| {
        rows.iter()
            .any(|r| r.form == "frame" && r.idle == *i && r.argmax == 0)
    });
    println!(
        "C1: argmax at red under form frame for 1, 4 and 8 idle frames: {}",
        verdict(pass)
    );
    write_json(
        &out.join("c1.json"),
        &json!({"mean_cosine": cos, "rows": rows, "pass_idle": C1_PASS_IDLE, "pass": pass}),
    )?;
    Ok(pass)
}

pub fn memcheck(a: &MemcheckArgs) -> Result<()> {
    std::fs::create_dir_all(&a.out)?;
    mlx_rs::with_device(mlx_rs::Device::cpu(), || run(a))
}

/// Amendment 1: the rules in their order of preference, the passes and the idle frames swept.
pub const SWEEP_RULES: [MemoryWrite; 3] =
    [MemoryWrite::Delta, MemoryWrite::Exact, MemoryWrite::Centred];
pub const SWEEP_PASSES: [usize; 4] = [1, 2, 4, 8];
pub const SWEEP_IDLE: [usize; 5] = [1, 4, 8, 16, 32];
/// The idle frames a rule must decode right, with at most `SWEEP_MAX_PASSES` passes.
pub const SWEEP_PASS_IDLE: [usize; 3] = [1, 4, 8];
pub const SWEEP_MAX_PASSES: usize = 4;

/// Amendment 1's adoption: the first rule of [`SWEEP_RULES`] that decodes red under form
/// `frame` (C1's criterion) at every idle count of [`SWEEP_PASS_IDLE`] with at most
/// [`SWEEP_MAX_PASSES`] passes, with the smallest such number of passes.
pub fn sweep_adopt(rows: &[memory::AlgebraRow]) -> Option<(&'static str, usize)> {
    for rule in SWEEP_RULES {
        for n in SWEEP_PASSES
            .iter()
            .copied()
            .filter(|n| *n <= SWEEP_MAX_PASSES)
        {
            let ok = SWEEP_PASS_IDLE.iter().all(|i| {
                rows.iter().any(|r| {
                    r.rule == rule.name()
                        && r.passes == n
                        && r.form == "frame"
                        && r.idle == *i
                        && r.argmax == 0
                })
            });
            if ok {
                return Some((rule.name(), n));
            }
        }
    }
    None
}

fn sweep(out: &std::path::Path) -> Result<Option<(&'static str, usize)>> {
    let mut rules = vec![MemoryWrite::Average];
    rules.extend(SWEEP_RULES);
    let (cos, rows) = memory::algebra(&rules, &SWEEP_PASSES, &SWEEP_IDLE, 0)?;
    let adopted = sweep_adopt(&rows);
    let mut md = format!(
        "# Amendment 1: the write-rule sweep on C1's synthetic states\n\nd = 96, one head, \
         Wqk = Wv = I, word vectors drawn once with seed 0 (mean cosine between words {cos:.3}); \
         sleep every {} frames with N passes (the wake pass included). Each cell: the colour \
         decoded at the ask's door (R = red, right; G, B, Y wrong) and the margin, red's \
         projection minus the best wrong one's. `average` is v1's rule, for reference.\n",
        memory::SLEEP_EVERY
    );
    for form in ["frame", "token"] {
        md += &format!(
            "\n## Value form `{form}`\n\n| rule | N | {} |\n|---|---:|{}\n",
            SWEEP_IDLE.map(|i| format!("d = {i}")).join(" | "),
            "---|".repeat(SWEEP_IDLE.len())
        );
        for rule in &rules {
            for n in SWEEP_PASSES {
                let cells: Vec<String> = SWEEP_IDLE
                    .iter()
                    .map(|i| {
                        let r = rows
                            .iter()
                            .find(|r| {
                                r.rule == rule.name()
                                    && r.passes == n
                                    && r.form == form
                                    && r.idle == *i
                            })
                            .expect("a reading");
                        format!("{} {:+.3}", ["R", "G", "B", "Y"][r.argmax], r.margin)
                    })
                    .collect();
                md += &format!("| {} | {n} | {} |\n", rule.name(), cells.join(" | "));
            }
        }
    }
    md += &match adopted {
        Some((rule, n)) => format!(
            "\nAdopted (the first of delta, exact, centred to decode red under form `frame` at \
             d = 1, 4 and 8 with at most 4 passes): **{rule}**, N = **{n}**.\n"
        ),
        None => "\nNo rule decodes red at d = 1, 4 and 8 with at most 4 passes: v2 stops here \
                 (amendment 1).\n"
            .to_string(),
    };
    print!("{md}");
    std::fs::write(out.join("sweep.md"), &md)?;
    write_json(
        &out.join("sweep.json"),
        &json!({"mean_cosine": cos, "rows": rows, "order": SWEEP_RULES.map(|r| r.name()),
            "pass_idle": SWEEP_PASS_IDLE, "max_passes": SWEEP_MAX_PASSES,
            "adopted": adopted.map(|(r, n)| json!({"rule": r, "passes": n}))}),
    )?;
    Ok(adopted)
}

fn run(a: &MemcheckArgs) -> Result<()> {
    if a.sweep {
        sweep(&a.out)?;
        return Ok(());
    }
    let base_cfg = S2Config::load(a.base.join("config.json"))?;
    let base = params::load(&a.base.join("model.safetensors"))?;
    let tok = Tok::load(&a.base.join("tokenizer.json"))?;
    if a.grads_only {
        let forms = gradients(a, &base_cfg, &base, &tok)?;
        write_json(
            &a.out.join("grads.json"),
            &json!({"batches": a.batches, "batch": a.batch, "seed": a.seed, "pad_to": a.pad_to,
                "passes": 1, "qat": true, "forms": forms}),
        )?;
        return Ok(());
    }
    let c1_pass = c1(&a.out)?;
    if a.c1_only {
        return Ok(());
    }
    if a.episodes < 2 {
        bail!("--episodes needs at least 2");
    }

    // ---------------------------------------------------------------- C2
    let seeds = train_seeds(a.from_seed, a.episodes);
    let n_train = a.episodes / 2;
    println!(
        "C2: {} recall train episodes (seeds {}..={}), centroids from the first {n_train}",
        seeds.len(),
        seeds[0],
        seeds[seeds.len() - 1]
    );
    let mut forms = Vec::new();
    let mut binding = None;
    for form in FORMS {
        let cfg = form_cfg(&base_cfg, form, a.write)?;
        let raw = step0_params(&cfg, &base, a.seed)?;
        let p = prepare(&raw, &cfg, true)?.into_owned();
        params::eval(&p)?;
        let mut enc = Encoder::new(&tok, &cfg)?;
        let t0 = std::time::Instant::now();
        let played = play(&p, &cfg, &mut enc, &seeds, a.rows)?;
        let f = read_form(form, &played, n_train);
        println!(
            "  form {:5} (qk {}, values {}): accuracy {:.4} on {} asks of delays 1-8 | all {:.4} | by delay {} | key overlap {:.4} ({:.0}s)",
            f.form,
            f.memory_qk,
            f.memory_value,
            f.accuracy_1_8,
            f.asks_1_8,
            f.accuracy_all,
            f.by_bucket.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>().join(" "),
            f.key_overlap,
            t0.elapsed().as_secs_f64()
        );
        if binding.is_none() {
            // The base's own door-token hidden state at the show frames (any form: Wo = 0).
            let train: Vec<(Vec<f32>, usize)> = played
                .shows
                .iter()
                .filter(|s| s.0 < n_train)
                .map(|s| (s.2.clone(), s.1))
                .collect();
            let test: Vec<(&[f32], usize)> = played
                .shows
                .iter()
                .filter(|s| s.0 >= n_train)
                .map(|s| (s.2.as_slice(), s.1))
                .collect();
            binding = Some(nearest_centroid(&train, &test));
        }
        forms.push(f);
    }
    let (bind_acc, bind_n) = binding.unwrap_or((f64::NAN, 0));
    println!(
        "  the base's door-token hidden state at the show frames: colour accuracy {bind_acc:.4} on {bind_n} shows"
    );
    let adopted = c2_adopt(forms[0].accuracy_1_8, forms[1].accuracy_1_8);
    let c2_pass = adopted.is_some();
    println!(
        "C2: the better v2 form reads {:.4} on delays 1-8 (bar {C2_BAR}): {}{}",
        forms[0].accuracy_1_8.max(forms[1].accuracy_1_8),
        verdict(c2_pass),
        adopted.map_or(String::new(), |f| format!(", form {f} adopted"))
    );
    write_json(
        &a.out.join("c2.json"),
        &json!({"episodes": seeds.len(), "from_seed": a.from_seed, "first_seed": seeds[0],
            "last_seed": seeds[seeds.len() - 1], "train_episodes": n_train, "qat": true,
            "init_seed": a.seed, "write": a.write.name(), "forms": forms, "base_binding": {"accuracy": bind_acc, "shows": bind_n},
            "bar": C2_BAR, "tie": C2_TIE, "adopted": adopted, "pass": c2_pass}),
    )?;

    // ---------------------------------------------------------------- C3
    let c3 = gradients(a, &base_cfg, &base, &tok)?;
    let c3_pass = adopted.is_some_and(|f| {
        let r = &c3[f];
        r["mean_pairwise_cosine_wo"].as_f64().unwrap_or(f64::NAN) >= C3_BAR
            && r["other_mem_exactly_zero"].as_bool().unwrap_or(false)
    });
    match adopted {
        Some(f) => println!(
            "C3: form {f}: cosine {:+.4} (bar {C3_BAR}), other mem.* gradients exactly zero: {}: {}",
            c3[f]["mean_pairwise_cosine_wo"].as_f64().unwrap_or(f64::NAN),
            c3[f]["other_mem_exactly_zero"],
            verdict(c3_pass)
        ),
        None => println!("C3: no form adopted by C2: FAIL"),
    }
    write_json(
        &a.out.join("c3.json"),
        &json!({"batches": a.batches, "batch": a.batch, "seed": a.seed, "pad_to": a.pad_to,
            "passes": 1, "qat": true, "write": a.write.name(), "forms": c3, "adopted": adopted, "bar": C3_BAR, "pass": c3_pass}),
    )?;
    println!(
        "C1 {} | C2 {} | C3 {}",
        verdict(c1_pass),
        verdict(c2_pass),
        verdict(c3_pass)
    );
    Ok(())
}

/// C3's measures on the three forms: per form, the mean pairwise cosine of the step-0 gradients
/// of `Wo` over the batches, the largest gradient of the other memory weights, and the gradient
/// norm of every parameter group.
fn gradients(
    a: &MemcheckArgs,
    base_cfg: &S2Config,
    base: &ParamMap,
    tok: &Tok,
) -> Result<BTreeMap<&'static str, serde_json::Value>> {
    let cfg0 = form_cfg(base_cfg, FORMS[2], a.write)?;
    let sd = SleepData::load(std::slice::from_ref(&a.seq), None, tok, &cfg0)?;
    let mut rng = Rng::new(a.seed);
    let batches: Vec<_> = (0..a.batches)
        .map(|_| sd.sample(a.batch, &mut rng, a.pad_to).1)
        .collect();
    println!(
        "C3: {} recall training batches of {} sequences ({} train sequences), step 0, N = 1, QAT",
        batches.len(),
        a.batch,
        sd.games[0].1.train.len()
    );
    let how = Sleep {
        passes: 1,
        every: SLEEP_EVERY,
        full: false,
    };
    let mut c3 = BTreeMap::new();
    for form in FORMS {
        let cfg = form_cfg(base_cfg, form, a.write)?;
        let p = step0_params(&cfg, base, a.seed)?;
        let wv = exit_weight_vec(&cfg, None);
        let wts = Array::from_slice(&wv, &[wv.len() as i32]);
        let t0 = std::time::Instant::now();
        let (mut wo, mut norms, mut other_max) = (Vec::new(), Vec::new(), 0f32);
        let mut losses = Vec::new();
        for b in &batches {
            let f = |q: HashMap<Rc<str>, Array>, _: ()| -> mlx_rs::error::Result<Vec<Array>> {
                Ok(vec![memory::seq_loss(
                    &q, &cfg, &b.frames, &b.pauses, &wts, true, how,
                )?])
            };
            let (loss, grads) = keyed_value_and_grad(f)(p.clone(), ())?;
            losses.push(loss[0].item_exact::<f32>() as f64);
            let mut names: Vec<&Rc<str>> = grads.keys().collect();
            names.sort();
            let mut g_wo = Vec::new();
            for n in names {
                if !n.starts_with("mem.") {
                    continue;
                }
                let g = host_f32(&grads[n])?;
                if n.ends_with(".out.weight") {
                    g_wo.extend(g);
                } else {
                    other_max = g.iter().fold(other_max, |m, x| m.max(x.abs()));
                }
            }
            norms.push(group_norms(&grads)?);
            wo.push(g_wo);
        }
        let cos = mean_pairwise_cosine(&wo);
        let mut mean_norms = BTreeMap::new();
        for g in norms[0].keys() {
            let m = norms.iter().map(|n| n[g]).sum::<f64>() / norms.len() as f64;
            mean_norms.insert(g.clone(), m);
        }
        println!(
            "  form {:5}: mean pairwise cosine of dL/dWo {cos:+.4} | largest other mem.* gradient {other_max:e} | gradient norm per group {} | loss {:.4} ({:.0}s)",
            form.name,
            mean_norms
                .iter()
                .map(|(k, v)| format!("{k} {v:.3e}"))
                .collect::<Vec<_>>()
                .join(", "),
            losses.iter().sum::<f64>() / losses.len() as f64,
            t0.elapsed().as_secs_f64()
        );
        c3.insert(
            form.name,
            json!({"memory_qk": form.qk.name(), "memory_value": form.value.name(),
                "mean_pairwise_cosine_wo": cos, "other_mem_max_abs_grad": other_max,
                "other_mem_exactly_zero": other_max == 0.0, "group_norms_mean": mean_norms,
                "group_norms": norms, "losses": losses}),
        );
    }
    Ok(c3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c2_adopts_the_better_form_and_frame_on_a_near_tie() {
        assert_eq!(c2_adopt(0.30, 0.39), None);
        assert_eq!(c2_adopt(0.45, 0.30), Some("frame"));
        assert_eq!(c2_adopt(0.30, 0.45), Some("token"));
        assert_eq!(c2_adopt(0.45, 0.46), Some("frame"));
        assert_eq!(c2_adopt(0.45, 0.48), Some("token"));
        assert_eq!(c2_adopt(0.40, 0.20), Some("frame"));
    }

    #[test]
    fn the_checks_read_train_seeds_only() {
        let s = train_seeds(0, 40);
        assert_eq!(s.len(), 40);
        assert!(s.iter().all(|x| x % 20 < 18));
        assert_eq!(&s[..3], &[0, 1, 2]);
        assert_eq!(s[18], 20);
    }

    #[test]
    fn recall_frames_parse() {
        assert_eq!(
            parse("show b colour blue ask none noise 4").unwrap(),
            (Some(1), Some(2), None)
        );
        assert_eq!(
            parse("show none colour none ask c noise 40").unwrap(),
            (None, None, Some(2))
        );
        assert!(parse("cells 0101").is_err());
        let v = [vec![1.0f32, 0.0], vec![1.0, 0.1], vec![-1.0, 0.0]];
        let c = mean_pairwise_cosine(&v);
        assert!((c - (0.995 - 1.0 - 0.995) / 3.0).abs() < 1e-3, "{c}");
    }
}
