//! The data of s6-route v5 (`docs/s6-route5-prereg.md`, section 4, with its amendment 1): v4
//! with every drawn distractor within the right tool's surface cell (S1) and the hard
//! distractors the tools nearest to the request (S1′). v2's build folder is read, never written;
//! v4's build folder is read for its rewrites (`teacher/`) and its validation slates; everything
//! v5 writes goes to its own build folder.
//!
//! 1. `scripts/route-embed.py --requests` (the lead runs it): `<b5>/emb/potion-requests.npy`, the
//!    training and validation requests of v2's plan embedded by potion-base-2M (4.1).
//! 2. [`plan`] (`sky-data route v5-plan --v2 <v2 build> --v4 <v4 build> --out <b5>`): v2's split
//!    recomputed and checked; for every training and validation request the natural slate, the
//!    hard slate (the 16 candidates nearest to the request within the cell, on v2's stream
//!    `route/plan/<id>`), the replacement sequence of the false-negative filter, the removed copies
//!    (D5) and the renamed slate with its removed twin, each at its level; `v5_tools` and
//!    `v5_tools_masked` (seed 1238); the counts by group, kind, split and level, which
//!    `scripts/s6-route5-checks.py --plan` reads for C6; the items of the `false-negatives` job,
//!    staged in `<b5>/stage/false-negatives5.items.jsonl` (the build script moves them to
//!    `<b5>/teacher/` once C6 holds).
//! 3. The teacher (the lead runs it): `BUILD=<b5> scripts/route-teacher.sh false-negatives loop`.
//! 4. [`slates`] (`sky-data route v5-slates --v2 <v2 build> --v4 <v4 build> --out <b5>`): the
//!    false-negative filter (the replacements from the slate's level, then the next; the slate's
//!    level raised), the dropped requests, the records of every kind (D1, D2, D4, D5, the field
//!    `cell_level` and the role of every option), the probe copies and the model's copies.
//!
//! Every step prints counts only.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sky_infer::route::{options_with_scores, Render};

use super::build::{
    self, answers_path, fn_id, stage, Able, Assigned, Ctx, FnItem, Kept, Loaded, Plan, RouteRecord,
    NEAREST, NONE_SHARE, REMOVED_SHARE,
};
use super::cluster::{self, Matrix};
use super::slates::{self, Slate};
use super::v4::{self, read_or_empty, refused_ids, surface, Pools, Rewrites, V2};
use super::{
    add, bump, option_text, read_lines, sha256_file, sha256_hex, write_json, Counts, Tool,
    INSTRUCTION, INTENTS, NONE_OPTION, TOOLS, WHEN2CALL,
};
use crate::record::write_jsonl;
use crate::rng::Rng;

/// The seed of the validation slates `v5_tools` (4.3, O28).
pub const V5_VAL_SEED: u64 = 1238;
/// The name of the new validation slates (files `val/v5_tools*.jsonl`).
pub const V5_TOOLS: &str = "v5-tools";
/// The plan of v5 in its build folder.
pub const PLAN5: &str = "plan5.jsonl";
/// 4.3: a slate is built at the lowest level whose candidates hold at least this many tools
/// (route/tools and When2Call; amendment 1, decision 2).
pub const MIN_CANDIDATES: usize = 16;
/// Amendment 1, decision 2: for the intents, the slate's drawn distractors plus this many.
pub const INTENTS_SPARE: usize = 3;
/// The replacement sequence kept with each hard slate: at most six distractors the teacher can
/// name, and D5's extra.
pub const MORE: usize = 7;
/// 4.1: the request embeddings and their meta (`scripts/route-embed.py --requests`).
pub const REQ_EMB: &str = "emb/potion-requests.npy";
pub const REQ_META: &str = "emb/potion-requests.meta.json";
/// The false-negative items before C6 has held (amendment 1, decision 1).
pub const FN_STAGED: &str = "stage/false-negatives5.items.jsonl";
/// The training pool of each group, for N2's frequency classes (`scripts/s6-route5-checks.py`).
pub const POOLS5: &str = "stage/pools5.json";
/// C6's bar (4.4): the share of a judged cell's drawn slates at L5 or L6.
pub const C6_BAR: f64 = 0.10;
/// The cells judged (v4 4.4): at least this many tool-gold slates.
pub const MIN_JUDGED: usize = 200;

/// Amendment 1, decision 2: the minimum number of candidates of a slate of `group` that draws
/// `drawn` distractors (D5's extra included for a removed slate).
pub fn minimum(group: &str, drawn: usize) -> usize {
    if group == INTENTS {
        drawn + INTENTS_SPARE
    } else {
        MIN_CANDIDATES
    }
}

// ------------------------------------------------------------------------------------ D1, fast

const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

fn fnv(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// D1's draw for the tools of one slate: [`v4::rewrite_draw`]`(seed, request, kind, tool)`, the
/// label's hash carried over its common prefix (the same bits, a test says so).
#[derive(Clone, Copy, Debug)]
pub struct D1 {
    seed: u64,
    prefix: u64,
}

impl D1 {
    pub fn new(seed: u64, request: &str, kind: &str) -> Self {
        D1 {
            seed,
            prefix: fnv(
                FNV_OFFSET,
                format!("route/rewrite/{request}/{kind}/").as_bytes(),
            ),
        }
    }

    pub fn draw(&self, tool: &str) -> bool {
        let h = crate::rng::mix64(fnv(self.prefix, tool.as_bytes()));
        Rng::new(self.seed ^ h).f64() < v4::REWRITE_P
    }
}

// ------------------------------------------------------------------------------ the cell (S1)

/// The candidates of one slate by level (4.3): the tools of the pool that pass the kind's
/// filter, each at the lowest level at which the text it would show matches the anchor's key.
/// Computed level by level, only as far as a slate needs.
pub struct Cell<'a> {
    table: &'a [u8],
    anchor: Option<usize>,
    /// The two keys a tool can show (its original text, its rewrite), as key indices.
    keys: &'a (dyn Fn(usize) -> (u16, u16) + 'a),
    /// D1's draw: whether the tool shows its rewrite.
    draw: &'a (dyn Fn(usize) -> bool + 'a),
    ok: &'a (dyn Fn(usize) -> bool + 'a),
    /// Tools by the lower of their two match levels, not yet placed.
    pending: Vec<Vec<(usize, u8, u8)>>,
    /// Tools (passing the filter) by the level of the text they show.
    by: Vec<Vec<usize>>,
    done: usize,
    mem: Vec<Option<Vec<usize>>>,
    /// The lowest level whose candidates reach the slate's minimum (L6 when none does, or when
    /// the slate has no anchor).
    pub level: u8,
}

impl<'a> Cell<'a> {
    pub fn new(
        pool: &[usize],
        anchor: Option<u16>,
        table: &'a [u8],
        keys: &'a (dyn Fn(usize) -> (u16, u16) + 'a),
        draw: &'a (dyn Fn(usize) -> bool + 'a),
        ok: &'a (dyn Fn(usize) -> bool + 'a),
        min: usize,
    ) -> Self {
        let mut pending = vec![Vec::new(); surface::L6 as usize];
        if let Some(a) = anchor {
            let row = &table[a as usize * surface::KEYS..(a as usize + 1) * surface::KEYS];
            for &t in pool {
                let (ko, kr) = keys(t);
                let (mo, mr) = (row[ko as usize], row[kr as usize]);
                let lo = mo.min(mr);
                if lo < surface::L6 {
                    pending[lo as usize].push((t, mo, mr));
                }
            }
        }
        let mut c = Cell {
            table,
            anchor: anchor.map(|a| a as usize),
            keys,
            draw,
            ok,
            pending,
            by: vec![Vec::new(); surface::L6 as usize],
            done: 0,
            mem: vec![None; surface::L6 as usize],
            level: surface::L6,
        };
        c.level = c.level_for(min);
        c
    }

    fn extend(&mut self, upto: u8) {
        while self.done <= upto as usize && self.done < surface::L6 as usize {
            for &(t, mo, mr) in &self.pending[self.done] {
                let s = if mo == mr || !(self.draw)(t) { mo } else { mr };
                if s < surface::L6 && (self.ok)(t) {
                    self.by[s as usize].push(t);
                }
            }
            self.done += 1;
        }
    }

    /// The candidates at `level` or below.
    pub fn count(&mut self, level: u8) -> usize {
        if level >= surface::L6 {
            return usize::MAX;
        }
        self.extend(level);
        self.by[..=level as usize].iter().map(Vec::len).sum()
    }

    /// The lowest level whose candidates hold at least `min` tools.
    pub fn level_for(&mut self, min: usize) -> u8 {
        if self.anchor.is_none() {
            return surface::L6;
        }
        (0..surface::L6)
            .find(|&l| self.count(l) >= min)
            .unwrap_or(surface::L6)
    }

    /// The candidates at `level` or below, in pool order (`level` below L6).
    pub fn members(&mut self, level: u8) -> Vec<usize> {
        let l = level.min(surface::L6 - 1) as usize;
        if let Some(m) = &self.mem[l] {
            return m.clone();
        }
        self.extend(l as u8);
        let mut v: Vec<usize> = self.by[..=l].iter().flatten().copied().collect();
        v.sort_unstable();
        self.mem[l] = Some(v.clone());
        v
    }

    /// The level of the text `t` shows (L6 when it matches nowhere before).
    pub fn level_of(&self, t: usize) -> u8 {
        let Some(a) = self.anchor else {
            return surface::L6;
        };
        let (ko, kr) = (self.keys)(t);
        let row = &self.table[a * surface::KEYS..(a + 1) * surface::KEYS];
        let (mo, mr) = (row[ko as usize], row[kr as usize]);
        if mo == mr || !(self.draw)(t) {
            mo
        } else {
            mr
        }
    }
}

/// The random fill of a slate (4.3, amendment 1): `s` filled to `want` from the cell's random
/// fill at `from` (the candidates of the level, those of the request's source first, D7), then
/// from each next level once a level is used up; at L6, D7's layers (the pool's tools of the
/// source, then the pool). Returns the level each added tool came from.
#[allow(clippy::too_many_arguments)]
pub fn fill_levels(
    s: &mut Vec<usize>,
    want: usize,
    from: u8,
    cell: &mut Cell,
    src: &[usize],
    pool: &[usize],
    rng: &mut Rng,
    ok: &dyn Fn(usize) -> bool,
) -> Vec<u8> {
    let mut levels = Vec::new();
    for lv in from..=surface::L6 {
        if s.len() >= want {
            break;
        }
        let before = s.len();
        if lv < surface::L6 {
            let m = cell.members(lv);
            let ms: Vec<usize> = m
                .iter()
                .copied()
                .filter(|t| src.binary_search(t).is_ok())
                .collect();
            slates::fill_layers(s, want, &[&ms, &m], rng, ok);
        } else {
            slates::fill_layers(s, want, &[src, pool], rng, ok);
        }
        levels.extend(std::iter::repeat_n(lv, s.len() - before));
    }
    levels
}

// ---------------------------------------------------------------------------------- the plan

/// One slate of the plan: its tools in the order shown, whether each is a drawn distractor (not
/// the right tool, not one of the request's own tools), and the level it is built at (4.3),
/// raised to the level of any distractor drawn from a later one.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct Planned {
    pub tools: Vec<String>,
    pub drawn: Vec<bool>,
    pub level: u8,
}

impl Planned {
    pub fn is_drawn(&self) -> bool {
        self.drawn.iter().any(|&d| d)
    }
}

/// One training or validation request of v5 and its slates before the teacher (tool ids).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Plan5 {
    pub id: String,
    pub group: String,
    pub source: String,
    pub split: String,
    pub lang: String,
    pub user: String,
    pub kind: String,
    pub gold: Option<String>,
    pub natural: Planned,
    /// D5: the natural slate with the gold replaced by one more distractor of its rule.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub natural_removed: Option<Planned>,
    /// The hard slate before the false-negative filter.
    pub hard: Planned,
    /// The level the hard slate's removed copy needs by its own minimum (amendment 1).
    pub hard_removed_min: u8,
    /// The replacement sequence of the hard slate: the rest of the request's 16 nearest tools
    /// within the cell, then the cell's random fill at the slate's level, then at the next ones;
    /// each with the level it comes from. The false-negative replacements, then D5's extra, take
    /// from it in order.
    pub hard_more: Vec<String>,
    pub hard_more_levels: Vec<u8>,
    /// D3: the renamed slate by the test-slate rule and its removed twin.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub renamed: Option<Planned>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub renamed_removed: Option<Planned>,
    /// The id of the hard slate's false-negative item (v2's `fn_id`).
    pub fn_id: String,
}

pub struct PlanOptions {
    /// v2's build folder (read only).
    pub v2: PathBuf,
    /// v4's build folder (read only: its rewrites and `v4_tools`).
    pub v4: PathBuf,
    /// v5's build folder.
    pub out: PathBuf,
    /// v2's seed.
    pub seed: u64,
}

/// The two keys each tool can show, as shown in real names and under `tool_<k>` names.
struct Keys {
    orig: Vec<u16>,
    rw: Vec<u16>,
    orig_m: Vec<u16>,
    rw_m: Vec<u16>,
}

/// What every request's plan reads.
struct World<'a> {
    l: &'a Loaded,
    a: &'a Assigned,
    seed: u64,
    keys: Keys,
    table: Vec<u8>,
    /// The pool's tools of each source, by (pool, source).
    by_source: HashMap<(String, String), Vec<usize>>,
    /// potion-base-2M: the tools' rows of v2's `emb/potion.npy`, then the requests' rows.
    comb: Matrix,
    tool_rows: usize,
    tool_of_row: HashMap<usize, usize>,
}

/// The sha256 of v2's plan and of its ids, which the request embeddings must have been written
/// for.
fn plan_ids_sha256(plans: &[Plan]) -> String {
    let ids: Vec<&str> = plans.iter().map(|p| p.id.as_str()).collect();
    sha256_hex(ids.join("\n").as_bytes())
}

/// 4.1: the request embeddings, checked against v2's plan (one row per request, in its order,
/// training and validation requests only).
fn load_requests(out: &Path, v2: &Path, plans: &[Plan], dim: usize) -> Result<Matrix> {
    let meta = super::read_json(&out.join(REQ_META)).with_context(|| {
        format!(
            "no request embeddings: scripts/route-embed.py --requests --v2 {} --build {}",
            v2.display(),
            out.display()
        )
    })?;
    let plan_path = stage(v2).join("plan.jsonl");
    if meta["source_sha256"].as_str() != Some(sha256_file(&plan_path)?.as_str()) {
        bail!(
            "{REQ_META} was not written for {}: run scripts/route-embed.py --requests again",
            plan_path.display()
        );
    }
    if meta["ids_sha256"].as_str() != Some(plan_ids_sha256(plans).as_str()) {
        bail!("{REQ_META}: the embedded requests are not v2's plan, in its order");
    }
    let m = Matrix::read_npy(&out.join(REQ_EMB))?;
    if m.rows != plans.len() || m.dim != dim {
        bail!(
            "{REQ_EMB}: {} rows of {} for {} requests of {dim}",
            m.rows,
            m.dim,
            plans.len()
        );
    }
    if plans.iter().any(|p| p.split != "train" && p.split != "val") {
        bail!("v2's plan holds a request that is not of training or validation");
    }
    Ok(m)
}

/// The rewrites of v2 and v4, every tool of `want` answered or refused (4.1: every tool of the
/// pools has one).
fn rewrites(v2: &Path, v4: &Path, want: &[usize], l: &Loaded) -> Result<Rewrites> {
    let rw = Rewrites::load(v2, v4)?;
    let missing = want
        .iter()
        .filter(|&&t| {
            let id = &l.tools[t].id;
            !rw.text.contains_key(id) && !rw.refused.contains(id)
        })
        .count();
    if missing > 0 {
        bail!("{missing} tools of the pools have no rewrite in v2's or v4's teacher answers (4.1 says every one has)");
    }
    Ok(rw)
}

/// The pools' tools and the intents, sorted: the catalogue of the cut points (4.3).
fn catalogue(pools: &Pools) -> Vec<usize> {
    let mut all: Vec<usize> = pools.train.iter().chain(&pools.val).copied().collect();
    all.extend(pools.intents.values().flatten().copied());
    all.sort_unstable();
    all.dedup();
    all
}

/// 4.3: the cut points over every text a training or validation slate can show (the original
/// and the rewrite of every tool of the catalogue), and the keys of every tool.
fn keys_of(l: &Loaded, rw: &Rewrites, cat: &[usize]) -> Result<(surface::Cuts, Keys)> {
    let mut texts: Vec<(&str, &str)> = Vec::with_capacity(2 * cat.len());
    for &t in cat {
        let tool = &l.tools[t];
        texts.push((&tool.name, &tool.desc));
        if let Some(d) = rw.text.get(&tool.id) {
            texts.push((&tool.name, d));
        }
    }
    let cuts = surface::Cuts::new(texts);
    // the renamed kind shows `tool_<k>`, k in 1..=99: F1 snake, F2 the class of 6 or 7
    // characters; the class must not depend on k, which is drawn after the slate
    let q6 = surface::quintile(&cuts.names, 6);
    let q7 = surface::quintile(&cuts.names, 7);
    if q6 != q7 {
        bail!("the names tool_1 and tool_10 fall in different length classes ({q6}, {q7}): the renamed kind's F2 is not known before its names are drawn (4.3)");
    }
    let masked = |k: surface::Key| -> u16 {
        surface::Key {
            style: 0,
            name_q: q6 as u8,
            ..k
        }
        .index() as u16
    };
    let n = l.tools.len();
    let mut keys = Keys {
        orig: vec![0; n],
        rw: vec![0; n],
        orig_m: vec![0; n],
        rw_m: vec![0; n],
    };
    for (t, tool) in l.tools.iter().enumerate() {
        let ko = cuts.key(&tool.name, &tool.desc);
        let kr = rw
            .text
            .get(&tool.id)
            .map_or(ko, |d| cuts.key(&tool.name, d));
        keys.orig[t] = ko.index() as u16;
        keys.rw[t] = kr.index() as u16;
        keys.orig_m[t] = masked(ko);
        keys.rw_m[t] = masked(kr);
    }
    Ok((cuts, keys))
}

/// What one request's plan gives.
struct One {
    plan: Plan5,
    counts: Counts,
}

/// The text of a slate's options: the source's (the validation slates), D1's draw in real names,
/// or D1's draw under `tool_<k>` names (the renamed kind).
#[derive(Clone, Copy)]
enum Shown {
    D1(D1),
    D1Masked(D1),
}

impl World<'_> {
    fn key_fn(&self, shown: Shown) -> impl Fn(usize) -> (u16, u16) + '_ {
        move |t| match shown {
            Shown::D1(_) => (self.keys.orig[t], self.keys.rw[t]),
            Shown::D1Masked(_) => (self.keys.orig_m[t], self.keys.rw_m[t]),
        }
    }

    fn draw_fn(&self, shown: Shown) -> impl Fn(usize) -> bool + '_ {
        move |t| match shown {
            Shown::D1(d) | Shown::D1Masked(d) => d.draw(&self.l.tools[t].id),
        }
    }

    /// The anchor's key: the text the anchor tool would show in this slate.
    fn anchor_key(&self, anchor: Option<usize>, shown: Shown) -> Option<u16> {
        let t = anchor?;
        let (ko, kr) = self.key_fn(shown)(t);
        Some(if ko == kr || !self.draw_fn(shown)(t) {
            ko
        } else {
            kr
        })
    }

    fn source(&self, pool_name: &str, source: &str) -> &[usize] {
        self.by_source
            .get(&(pool_name.to_string(), source.to_string()))
            .map_or(&[], Vec::as_slice)
    }

    #[allow(clippy::too_many_lines)]
    fn plan_one(&self, p: &Plan, k: &Kept, pools: &Pools, req_row: usize) -> Result<One> {
        let l = self.l;
        let a = self.a;
        let cx = Ctx { l, seed: self.seed };
        let mut counts = Counts::new();
        let gold = p.gold.as_ref().map(|g| l.index[g]);
        if k.split != p.split || k.gold != gold {
            bail!("{}: v2's plan and the recomputed split differ", p.id);
        }
        let (gs, sp) = (&p.group, &p.split);
        let (pool_name, pool) = pools.of(k);
        let src = self.source(&pool_name, &k.r.source);
        let same_cluster = |t: usize| match gold {
            Some(g) => t == g || (a.cluster_of[g].is_some() && a.cluster_of[t] == a.cluster_of[g]),
            None => false,
        };
        let ok = |t: usize| {
            !k.exclude.contains(&t)
                && !same_cluster(t)
                && (k.r.group == INTENTS || a.allowed(&k.r.group, &k.split, t))
        };
        // the anchor: the right tool, else the request's first own tool (4.3)
        let anchor = gold.or_else(|| k.own.first().copied());
        let ids = |v: &[usize]| v.iter().map(|&t| l.tools[t].id.clone()).collect::<Vec<_>>();
        let own_ok = k.own.iter().filter(|&&t| Some(t) != gold && ok(t)).count();
        let g1 = gold.is_some() as usize;

        // ---- the natural slate: v2's size, the request's own tools, the cell's random fill
        let n = p.natural.len();
        let nat_drawn = n.saturating_sub(g1 + own_ok.min(n.saturating_sub(g1)));
        let sh = Shown::D1(D1::new(self.seed, &p.id, "natural"));
        let (kf, df) = (self.key_fn(sh), self.draw_fn(sh));
        let mut nc = Cell::new(
            pool,
            self.anchor_key(anchor, sh),
            &self.table,
            &kf,
            &df,
            &ok,
            minimum(gs, nat_drawn),
        );
        let mut rng = cx.rng(&format!("route5/natural/{}", p.id));
        let mut s: Vec<usize> = gold.into_iter().collect();
        for &t in &k.own {
            if s.len() >= n {
                break;
            }
            if Some(t) != gold && !s.contains(&t) && ok(t) {
                s.push(t);
            }
        }
        let fixed = s.len();
        let lv = fill_levels(&mut s, n, nc.level, &mut nc, src, pool, &mut rng, &ok);
        s.truncate(n.max(g1));
        // a drawn distractor is one the fill drew (not the gold, not an own tool placed first)
        let (s, nat_flags) = shuffled(s, fixed, &mut rng);
        let natural = Planned {
            drawn: nat_flags.clone(),
            level: lv.iter().copied().fold(nc.level, u8::max),
            tools: ids(&s),
        };
        if s.len() != n {
            bump(
                &mut counts,
                format!("{gs} {sp}: natural slates of another size than v2's"),
            );
        }
        // D5: one more distractor of the natural rule in the gold's place, in the gold's cell, on
        // the natural slate's text, drawn from the natural slate's level (amendment 2, decision
        // 1); the copy's level is also the one its own minimum needs (amendment 1: the minimum
        // counts the extra)
        let natural_removed = match gold {
            None => None,
            Some(g) => {
                let lr = nc.level_for(minimum(gs, nat_drawn + 1)).max(natural.level);
                let mut t: Vec<usize> = s.iter().copied().filter(|&x| x != g).collect();
                let want = t.len() + 1;
                let okg = |x: usize| x != g && ok(x);
                let from = natural.level;
                let le = fill_levels(&mut t, want, from, &mut nc, src, pool, &mut rng, &okg);
                let (mut v, mut flags) = (s.clone(), nat_flags.clone());
                let gp = v.iter().position(|&x| x == g).unwrap();
                if t.len() == want {
                    v[gp] = *t.last().unwrap();
                    flags[gp] = true;
                } else {
                    // as v4: without a distractor left, the copy is the slate without its gold
                    bump(
                        &mut counts,
                        format!("{gs} {sp}: no distractor left for a natural removed copy"),
                    );
                    v.remove(gp);
                    flags.remove(gp);
                }
                (!v.is_empty()).then(|| Planned {
                    drawn: flags,
                    level: le.iter().copied().fold(lr, u8::max),
                    tools: ids(&v),
                })
            }
        };

        // ---- the hard slate: v2's size and stream; the right tool, or for a request without a
        // tool gold its own tools first (exempt: amendment 2, decision 3); then the 16 candidates
        // nearest to the request within the cell, the right tool excluded (S1′)
        let size = p.hard.len();
        let base: Vec<usize> = match gold {
            Some(g) => vec![g],
            None => {
                let mut b: Vec<usize> = Vec::new();
                for &t in &k.own {
                    if b.len() >= size {
                        break;
                    }
                    if !b.contains(&t) && ok(t) {
                        b.push(t);
                    }
                }
                b
            }
        };
        let nb = base.len();
        let need = size.saturating_sub(nb);
        let sh = Shown::D1(D1::new(self.seed, &p.id, "hard"));
        let (kf, df) = (self.key_fn(sh), self.draw_fn(sh));
        let mut hc = Cell::new(
            pool,
            self.anchor_key(anchor, sh),
            &self.table,
            &kf,
            &df,
            &ok,
            minimum(gs, need),
        );
        let hard_removed_min = hc.level_for(minimum(gs, need + 1));
        let lh = hc.level;
        let cands: Vec<usize> = if lh < surface::L6 {
            hc.members(lh)
        } else {
            pool.iter().copied().filter(|&t| ok(t)).collect()
        };
        let rows: Vec<usize> = cands.iter().filter_map(|&t| l.row[t]).collect();
        let near: Vec<usize> =
            cluster::nearest(&self.comb, self.tool_rows + req_row, &rows, NEAREST, |_| {
                false
            })
            .into_iter()
            .map(|r| self.tool_of_row[&r])
            .collect();
        let mut rng = cx.rng(&format!("route/plan/{}", p.id));
        // v2's draws of the two sizes, kept so that the stream is v2's from here on
        let _ = (
            slates::draw_options(&mut rng),
            slates::draw_options(&mut rng),
        );
        let mut seq: Vec<usize> = base.clone();
        seq.extend(near.iter().copied().filter(|t| !base.contains(t)));
        let mut seq_lv: Vec<u8> = vec![lh; seq.len() - nb];
        let want = nb + need + MORE;
        if seq.len() < want {
            if lh < surface::L6 {
                bump(
                    &mut counts,
                    format!("{gs} {sp}: hard sequences completed from the cell's random fill"),
                );
            }
            seq_lv.extend(fill_levels(
                &mut seq, want, lh, &mut hc, src, pool, &mut rng, &ok,
            ));
        }
        seq.truncate(want);
        seq_lv.truncate(want - nb);
        let body = &seq[nb..];
        let take = need.min(body.len());
        let hard: Vec<usize> = base
            .iter()
            .copied()
            .chain(body[..take].iter().copied())
            .collect();
        let hard_level = seq_lv[..take].iter().copied().fold(lh, u8::max);
        if hard.len() != p.hard.len() {
            bump(
                &mut counts,
                format!("{gs} {sp}: hard slates of another size than v2's"),
            );
        }
        let hard_more = ids(&body[take..]);
        let hard_more_levels = seq_lv[take..].to_vec();
        let (hard, hard_flags) = shuffled(hard, nb, &mut rng);
        let hard_p = Planned {
            drawn: hard_flags,
            level: hard_level,
            tools: ids(&hard),
        };

        // ---- D3: the renamed slate by the test-slate rule, the fill of the natural kind, in the
        // cell of the text under `tool_<k>` names; its removed twin
        let (renamed, renamed_removed) = match gold {
            None => (None, None),
            Some(_) => {
                let own_p: Vec<usize> = k
                    .own
                    .iter()
                    .copied()
                    .filter(|t| pool.binary_search(t).is_ok())
                    .collect();
                let tok = v4::test_ok(&cx, k);
                let own_t = own_p.iter().filter(|&&t| Some(t) != gold && tok(t)).count();
                let drawn = slates::TEST_TOOLS - 1 - own_t.min(slates::TEST_TOOLS - 1);
                let sh = Shown::D1Masked(D1::new(self.seed, &p.id, "renamed"));
                let (kf, df) = (self.key_fn(sh), self.draw_fn(sh));
                let mut rc = Cell::new(
                    pool,
                    self.anchor_key(anchor, sh),
                    &self.table,
                    &kf,
                    &df,
                    &tok,
                    minimum(gs, drawn),
                );
                let lr = rc.level_for(minimum(gs, drawn + 1));
                let rng = cx.rng(&format!("route5/renamed-slate/{}", p.id));
                let (present, lp, fp) = test_like(
                    gold,
                    &own_p,
                    &mut rc,
                    None,
                    src,
                    pool,
                    &mut rng.clone(),
                    &tok,
                );
                let (twin, lt, ft) = test_like(
                    gold,
                    &own_p,
                    &mut rc,
                    Some(lr),
                    src,
                    pool,
                    &mut rng.clone(),
                    &tok,
                );
                if present.tools.len() != slates::TEST_TOOLS
                    || twin.tools.len() != slates::TEST_TOOLS
                {
                    bump(
                        &mut counts,
                        format!("{gs} {sp}: renamed slates of fewer than 7 tools"),
                    );
                }
                let pl = |s: &Slate, level: u8, drawn: Vec<bool>| Planned {
                    drawn,
                    level,
                    tools: ids(&s.tools),
                };
                (Some(pl(&present, lp, fp)), Some(pl(&twin, lt, ft)))
            }
        };
        let fid = fn_id(
            &Plan {
                hard: hard_p.tools.clone(),
                ..p.clone()
            },
            &l.tools,
            &l.index,
        );
        bump(&mut counts, format!("{gs} {sp}: requests"));
        Ok(One {
            plan: Plan5 {
                id: p.id.clone(),
                group: p.group.clone(),
                source: p.source.clone(),
                split: p.split.clone(),
                lang: p.lang.clone(),
                user: p.user.clone(),
                kind: p.kind.clone(),
                gold: p.gold.clone(),
                natural,
                natural_removed,
                hard: hard_p,
                hard_removed_min,
                hard_more,
                hard_more_levels,
                renamed,
                renamed_removed,
                fn_id: fid,
            },
            counts,
        })
    }
}

/// [`slates::test_slate_layers`] by the cell (4.3): the right tool, the own tools `ok` accepts
/// (exempt), then the cell's random fill to 7 tools from the cell's level up; a removed slate
/// (`removed`: the level its own minimum needs) drops the right tool and takes one more from
/// that level up. Two calls on clones of one stream give the same tools before the removal.
/// Returns the slate, its level and which of its tools the fill drew.
#[allow(clippy::too_many_arguments)]
pub fn test_like(
    gold: Option<usize>,
    own: &[usize],
    cell: &mut Cell,
    removed: Option<u8>,
    src: &[usize],
    pool: &[usize],
    rng: &mut Rng,
    ok: &dyn Fn(usize) -> bool,
) -> (Slate, u8, Vec<bool>) {
    let mut s: Vec<usize> = gold.into_iter().collect();
    for &t in own {
        if s.len() >= slates::TEST_TOOLS {
            break;
        }
        if Some(t) != gold && !s.contains(&t) && ok(t) {
            s.push(t);
        }
    }
    let mut fixed = s.len();
    let base = cell.level;
    let mut level = fill_levels(&mut s, slates::TEST_TOOLS, base, cell, src, pool, rng, ok)
        .into_iter()
        .fold(base, u8::max);
    if let (Some(lr), Some(g)) = (removed, gold) {
        s.retain(|&t| t != g);
        fixed -= 1;
        let okg = |t: usize| t != g && ok(t);
        // the extra from the slate's own level, on its text (amendment 2, decision 1); the
        // twin's level is also the one its own minimum needs
        let from = level;
        level = fill_levels(&mut s, slates::TEST_TOOLS, from, cell, src, pool, rng, &okg)
            .into_iter()
            .fold(lr.max(level), u8::max);
    }
    let (s, drawn) = shuffled(s, fixed, rng);
    let slate = Slate::new(s, if removed.is_some() { None } else { gold });
    (slate, level, drawn)
}

/// `s` shuffled (the permutation of `rng.shuffle(&mut s)`), with whether each tool was drawn:
/// the tools from position `fixed` on (the gold and the own tools come first).
fn shuffled(s: Vec<usize>, fixed: usize, rng: &mut Rng) -> (Vec<usize>, Vec<bool>) {
    let mut v: Vec<(usize, bool)> = s
        .into_iter()
        .enumerate()
        .map(|(i, t)| (t, i >= fixed))
        .collect();
    rng.shuffle(&mut v);
    v.into_iter().unzip()
}

/// One record of v5: v4's fields, the slate's level and the role of each tool option.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Record5 {
    #[serde(flatten)]
    pub r: RouteRecord,
    /// 4.3: the level the slate was built at (0 to 6; 6 is outside the cell).
    pub cell_level: u8,
    /// Per tool option: `right`, `exempt` (one of the request's own tools, shown whatever its
    /// cell) or `drawn` (a drawn distractor, within the cell).
    pub roles: Vec<String>,
    /// A removed copy (D5): the kind of the slate it comes from (`natural` or `hard`), whose
    /// text it shows (amendment 2, decision 1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
}

pub const RIGHT: &str = "right";
pub const EXEMPT: &str = "exempt";
pub const DRAWN: &str = "drawn";

/// The role of each tool option: the right tool, an exempt tool or a drawn distractor.
pub fn roles(drawn: &[bool], gold: Option<usize>) -> Vec<String> {
    drawn
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            if Some(i) == gold {
                RIGHT
            } else if d {
                DRAWN
            } else {
                EXEMPT
            }
            .to_string()
        })
        .collect()
}

/// The C6 counts of one cell: slates, tool-gold slates, drawn slates, and both by level.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct LevelCounts {
    pub slates: usize,
    #[serde(rename = "tool-gold slates")]
    pub tool_gold: usize,
    #[serde(rename = "drawn slates")]
    pub drawn: usize,
    #[serde(rename = "slates by level")]
    pub by_level: [usize; 7],
    #[serde(rename = "drawn slates by level")]
    pub drawn_by_level: [usize; 7],
}

impl LevelCounts {
    pub fn add(&mut self, level: u8, drawn: bool, tool_gold: bool) {
        self.slates += 1;
        self.tool_gold += tool_gold as usize;
        self.by_level[level as usize] += 1;
        if drawn {
            self.drawn += 1;
            self.drawn_by_level[level as usize] += 1;
        }
    }

    /// C6: the share of the drawn slates at L5 or L6.
    pub fn share_l5_l6(&self) -> f64 {
        (self.drawn_by_level[5] + self.drawn_by_level[6]) as f64 / self.drawn.max(1) as f64
    }

    /// C6's cells (amendment 2, decision 2): at least 200 drawn slates, the removed cells
    /// included.
    pub fn judged(&self) -> bool {
        self.drawn >= MIN_JUDGED
    }
}

/// The C6 counts as JSON: one entry per cell, with the share and the verdict at the bar.
fn c6_json(cells: &BTreeMap<String, LevelCounts>) -> Value {
    let v: Vec<Value> = cells
        .iter()
        .map(|(name, c)| {
            json!({"cell": name, "counts": c, "judged": c.judged(),
                   "share of the drawn slates at L5 or L6": c.share_l5_l6(),
                   "share of the drawn slates at L6 (outside the cell)": c.drawn_by_level[6] as f64 / c.drawn.max(1) as f64,
                   "above the bar": c.share_l5_l6() > C6_BAR})
        })
        .collect();
    json!(v)
}

// ---------------------------------------------------------------------- the slates of a request

/// One request's slates as they are written: the planned ones, or the ones the false-negative
/// filter gives (4.2).
#[derive(Clone, Debug)]
struct Final {
    natural: Planned,
    natural_removed: Option<Planned>,
    hard: Planned,
    hard_removed: Option<Planned>,
    renamed: Option<Planned>,
    renamed_removed: Option<Planned>,
}

/// The teacher's word on a hard slate.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    /// The plan before the teacher: no filter.
    Planned,
    /// The tools (1-based positions) the teacher says can handle the request.
    Able(Vec<usize>),
    /// The moderation refused the item: the hard slate unfiltered (counted).
    Refused,
}

/// 4.2 and amendment 1: the hard slate after the teacher's answer, its removed copy (D5's extra
/// the next tool of the replacement sequence), and whether the request is dropped (the teacher
/// does not name the gold). A replacement takes the next tool of the sequence that is not in the
/// slate and was not named; the slate's level becomes the highest level any of its drawn
/// distractors came from.
pub fn filter_hard(
    p: &Plan5,
    verdict: &Verdict,
    counts: &mut Counts,
) -> Option<(Planned, Option<Planned>)> {
    let (gs, sp) = (&p.group, &p.split);
    let gold = p
        .gold
        .as_ref()
        .and_then(|g| p.hard.tools.iter().position(|t| t == g));
    let mut hard = p.hard.clone();
    let mut named: HashSet<String> = HashSet::new();
    let mut more = p.hard_more.iter().zip(&p.hard_more_levels);
    let mut next = |hard: &Planned, named: &HashSet<String>| {
        more.by_ref()
            .find(|(t, _)| !hard.tools.contains(t) && !named.contains(*t))
            .map(|(t, &lv)| (t.clone(), lv))
    };
    match verdict {
        Verdict::Planned => {}
        Verdict::Refused => bump(
            counts,
            format!("{gs} {sp}: refused by moderation, hard slate unfiltered"),
        ),
        Verdict::Able(ans) => {
            if let Some(gp) = gold {
                if !ans.contains(&(gp + 1)) {
                    bump(
                        counts,
                        format!("{gs} {sp}: dropped, the teacher says the gold cannot"),
                    );
                    return None;
                }
            }
            for i in 0..hard.tools.len() {
                if Some(i) == gold || !ans.contains(&(i + 1)) {
                    continue;
                }
                named.insert(hard.tools[i].clone());
            }
            for i in 0..hard.tools.len() {
                if !named.contains(&hard.tools[i]) {
                    continue;
                }
                bump(counts, format!("{gs} {sp}: distractors replaced"));
                match next(&hard, &named) {
                    Some((t, lv)) => {
                        if lv > p.hard.level {
                            bump(
                                counts,
                                format!(
                                    "{gs} {sp}: replacements from a later level than the slate's"
                                ),
                            );
                        }
                        hard.tools[i] = t;
                        hard.drawn[i] = true;
                        hard.level = hard.level.max(lv);
                    }
                    None => bump(
                        counts,
                        format!("{gs} {sp}: no tool left to replace a distractor"),
                    ),
                }
            }
        }
    }
    let removed = gold.and_then(|gp| {
        let mut r = hard.clone();
        r.level = r.level.max(p.hard_removed_min);
        match next(&hard, &named) {
            Some((t, lv)) => {
                r.tools[gp] = t;
                r.drawn[gp] = true;
                r.level = r.level.max(lv);
            }
            None => {
                // as v4: without a distractor left, the copy is the slate without its gold
                bump(
                    counts,
                    format!("{gs} {sp}: no distractor left for a hard removed copy"),
                );
                r.tools.remove(gp);
                r.drawn.remove(gp);
            }
        }
        (!r.tools.is_empty()).then_some(r)
    });
    Some((hard, removed))
}

/// Which slate of a request a written slate is.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Which {
    Natural,
    Hard,
    NaturalRemoved,
    HardRemoved,
    Renamed,
    RenamedRemoved,
}

/// A slate chosen for the records: its request (an index of the finals), which slate, its
/// record kind and uid suffix.
#[derive(Clone, Debug)]
struct Pick {
    req: usize,
    which: Which,
    kind: &'static str,
    uid: &'static str,
}

impl Pick {
    /// The kind of the slate a removed copy comes from.
    fn from(&self) -> Option<&'static str> {
        match self.which {
            Which::NaturalRemoved => Some("natural"),
            Which::HardRemoved => Some("hard"),
            _ => None,
        }
    }

    fn planned<'a>(&self, f: &'a [Option<Final>]) -> &'a Planned {
        let f = f[self.req].as_ref().unwrap();
        match self.which {
            Which::Natural => &f.natural,
            Which::Hard => &f.hard,
            Which::NaturalRemoved => f.natural_removed.as_ref().unwrap(),
            Which::HardRemoved => f.hard_removed.as_ref().unwrap(),
            Which::Renamed => f.renamed.as_ref().unwrap(),
            Which::RenamedRemoved => f.renamed_removed.as_ref().unwrap(),
        }
    }
}

/// v4's choice of the written slates (`v4::slates`), on the slates of v5: per group and split,
/// the natural and hard slates of every request with a tool gold, one of the two removed copies
/// (alternately), the renamed slate or its removed twin (*none* on 20% of them, `removed_set`),
/// the no-call slates, then the share of *none* of v2 4.4 on the slates that are not renamed.
/// Returns, per (group, split), the chosen slates sorted by uid, and the counts.
fn choose(
    plans: &[Plan5],
    finals: &[Option<Final>],
    seed: u64,
    counts: &mut Counts,
) -> BTreeMap<(String, String), Vec<Pick>> {
    let mut tool_gold: BTreeMap<(String, String), usize> = BTreeMap::new();
    for (p, f) in plans.iter().zip(finals) {
        if p.gold.is_some() && f.is_some() {
            *tool_gold
                .entry((p.group.clone(), p.split.clone()))
                .or_default() += 1;
        }
    }
    let renamed_removed: BTreeMap<(String, String), HashSet<usize>> = tool_gold
        .iter()
        .map(|((g, s), &n)| {
            let mut rng = Rng::derived(seed, &format!("route5/renamed/removed/{g}/{s}"));
            (
                (g.clone(), s.clone()),
                slates::removed_set(n, NONE_SHARE, &mut rng),
            )
        })
        .collect();
    #[derive(Default)]
    struct Bucket {
        fixed: Vec<Pick>,
        removed: Vec<Pick>,
        spare: Vec<Pick>,
        nocall: Vec<Pick>,
        renamed: Vec<Pick>,
    }
    let mut buckets: BTreeMap<(String, String), Bucket> = BTreeMap::new();
    let mut seen: BTreeMap<(String, String), usize> = BTreeMap::new();
    let pick = |req, which, kind, uid| Pick {
        req,
        which,
        kind,
        uid,
    };
    for (i, (p, f)) in plans.iter().zip(finals).enumerate() {
        let Some(f) = f else { continue };
        let key = (p.group.clone(), p.split.clone());
        let b = buckets.entry(key.clone()).or_default();
        if p.gold.is_none() {
            b.nocall.push(pick(i, Which::Natural, "natural", "natural"));
            b.nocall.push(pick(i, Which::Hard, "hard", "hard"));
            continue;
        }
        b.fixed.push(pick(i, Which::Natural, "natural", "natural"));
        b.fixed.push(pick(i, Which::Hard, "hard", "hard"));
        let n = seen.entry(key.clone()).or_default();
        let (nat, hard) = (
            f.natural_removed.as_ref().map(|_| Which::NaturalRemoved),
            f.hard_removed.as_ref().map(|_| Which::HardRemoved),
        );
        let order = if (*n).is_multiple_of(2) {
            [(nat, false), (hard, true)]
        } else {
            [(hard, false), (nat, true)]
        };
        let j = *n;
        *n += 1;
        for (w, spare) in order {
            let Some(w) = w else {
                bump(
                    counts,
                    format!(
                        "{} {}: no removed slate (no distractor left)",
                        p.group, p.split
                    ),
                );
                continue;
            };
            if spare {
                b.spare.push(pick(i, w, "removed", "removed-2"));
            } else {
                b.removed.push(pick(i, w, "removed", "removed"));
            }
        }
        let none = renamed_removed[&key].contains(&j);
        let w = if none {
            Which::RenamedRemoved
        } else {
            Which::Renamed
        };
        b.renamed.push(pick(i, w, "renamed", "renamed"));
    }
    let mut out = BTreeMap::new();
    for ((group, split), mut b) in buckets {
        let t = b.fixed.len();
        let target = ((t as f64) * NONE_SHARE / (1.0 - NONE_SHARE)).round() as usize;
        let mut rng = Rng::derived(seed, &format!("route5/none-share/{group}/{split}"));
        let mut none: Vec<Pick> = Vec::new();
        if b.nocall.len() >= target {
            rng.shuffle(&mut b.nocall);
            add(
                counts,
                format!("{group} {split}: removed slates dropped"),
                b.removed.len(),
            );
            add(
                counts,
                format!("{group} {split}: no-call slates dropped"),
                b.nocall.len() - target,
            );
            b.nocall.truncate(target);
            none.extend(b.nocall);
        } else {
            let room = target - b.nocall.len();
            none.extend(b.nocall);
            if b.removed.len() >= room {
                rng.shuffle(&mut b.removed);
                add(
                    counts,
                    format!("{group} {split}: removed slates dropped"),
                    b.removed.len() - room,
                );
                b.removed.truncate(room);
                none.extend(b.removed);
            } else {
                let more = room - b.removed.len();
                none.extend(b.removed);
                rng.shuffle(&mut b.spare);
                add(
                    counts,
                    format!("{group} {split}: removed slates added"),
                    more.min(b.spare.len()),
                );
                none.extend(b.spare.into_iter().take(more));
            }
        }
        let mut all = b.fixed;
        all.extend(none);
        all.extend(b.renamed);
        all.sort_by_cached_key(|x| uid_of(&plans[x.req], x));
        out.insert((group, split), all);
    }
    out
}

/// The uid of a written slate, as v4 writes it.
fn uid_of(p: &Plan5, pick: &Pick) -> String {
    format!("{}/{}", p.id, pick.uid)
}

/// The C6 counts of the chosen slates, by cell (`records <group> <kind> <split>`).
fn level_counts(
    plans: &[Plan5],
    finals: &[Option<Final>],
    chosen: &BTreeMap<(String, String), Vec<Pick>>,
) -> BTreeMap<String, LevelCounts> {
    let mut cells: BTreeMap<String, LevelCounts> = BTreeMap::new();
    for ((group, split), picks) in chosen {
        for pk in picks {
            let pl = pk.planned(finals);
            let tool_gold = matches!(pk.which, Which::Natural | Which::Hard | Which::Renamed)
                && plans[pk.req].gold.is_some();
            cells
                .entry(format!("records {group} {} {split}", pk.kind))
                .or_default()
                .add(pl.level, pl.is_drawn(), tool_gold);
        }
    }
    cells
}

/// The C6 counts of a validation slate file.
fn file_counts(recs: &[Record5]) -> LevelCounts {
    let mut c = LevelCounts::default();
    for r in recs {
        let tg = r.r.target.last() != Some(&1.0);
        c.add(r.cell_level, r.roles.iter().any(|x| x == DRAWN), tg);
    }
    c
}

/// The plan's slates as the build would write them if the teacher dropped no request and named
/// no distractor.
fn planned_finals(plans: &[Plan5], counts: &mut Counts) -> Vec<Option<Final>> {
    plans
        .iter()
        .map(|p| {
            let (hard, hard_removed) = filter_hard(p, &Verdict::Planned, counts)?;
            Some(Final {
                natural: p.natural.clone(),
                natural_removed: p.natural_removed.clone(),
                hard,
                hard_removed,
                renamed: p.renamed.clone(),
                renamed_removed: p.renamed_removed.clone(),
            })
        })
        .collect()
}

/// Step 2 of v5: the plan, its counts for C6, `v5_tools` and the false-negative items.
#[allow(clippy::too_many_lines)]
pub fn plan(o: &PlanOptions) -> Result<Value> {
    if o.out == o.v2 || o.out == o.v4 {
        bail!(
            "v5 writes to its own folder, never to v2's or v4's ({})",
            o.out.display()
        );
    }
    let V2 { l, a, kept } = v4::load_v2(&o.v2, o.seed)?;
    let pools = Pools::new(&a, &kept);
    let cat = catalogue(&pools);
    let rw = rewrites(&o.v2, &o.v4, &cat, &l)?;
    let (cuts, keys) = keys_of(&l, &rw, &cat)?;
    let plans: Vec<Plan> = read_lines(&stage(&o.v2).join("plan.jsonl"))?;
    let by_id: HashMap<&str, &Kept> = kept.iter().map(|k| (k.r.id.as_str(), k)).collect();
    let reqs = load_requests(&o.out, &o.v2, &plans, l.potion.dim)?;
    // the tools' sources (D7), per pool
    let mut src_of: HashMap<String, HashSet<usize>> = HashMap::new();
    for r in &l.requests {
        let e = src_of.entry(r.source.clone()).or_default();
        for id in r.tools.iter().chain(r.gold.iter()).chain(r.exclude.iter()) {
            e.insert(l.index[id]);
        }
    }
    let mut by_source: HashMap<(String, String), Vec<usize>> = HashMap::new();
    let mut ks: Vec<&Kept> = Vec::with_capacity(plans.len());
    for p in &plans {
        let Some(k) = by_id.get(p.id.as_str()).copied() else {
            bail!(
                "v2's plan holds {} that the recomputed split does not keep",
                p.id
            );
        };
        let (pn, pool) = pools.of(k);
        let empty = HashSet::new();
        let s = src_of.get(&k.r.source).unwrap_or(&empty);
        by_source
            .entry((pn, k.r.source.clone()))
            .or_insert_with(|| pool.iter().copied().filter(|t| s.contains(t)).collect());
        ks.push(k);
    }
    for k in build::tool_reqs(&kept, "val") {
        let empty = HashSet::new();
        let s = src_of.get(&k.r.source).unwrap_or(&empty);
        by_source
            .entry(("tools/val".to_string(), k.r.source.clone()))
            .or_insert_with(|| {
                pools
                    .val
                    .iter()
                    .copied()
                    .filter(|t| s.contains(t))
                    .collect()
            });
    }
    let tool_rows = l.potion.rows;
    let mut data = l.potion.data.clone();
    data.extend_from_slice(&reqs.data);
    let comb = Matrix {
        rows: tool_rows + reqs.rows,
        dim: l.potion.dim,
        data,
    };
    let tool_of_row: HashMap<usize, usize> = l
        .row
        .iter()
        .enumerate()
        .filter_map(|(t, r)| r.map(|r| (r, t)))
        .collect();
    let world = World {
        l: &l,
        a: &a,
        seed: o.seed,
        keys,
        table: surface::level_table(),
        by_source,
        comb,
        tool_rows,
        tool_of_row,
    };
    // every request's plan, on threads (the result does not depend on their number)
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(16);
    let parts: Vec<Vec<(usize, Result<One>)>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let (world, plans, ks, pools) = (&world, &plans, &ks, &pools);
                s.spawn(move || {
                    (t..plans.len())
                        .step_by(threads)
                        .map(|i| (i, world.plan_one(&plans[i], ks[i], pools, i)))
                        .collect()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut ones: Vec<(usize, Result<One>)> = parts.into_iter().flatten().collect();
    ones.sort_by_key(|(i, _)| *i);
    let mut counts = Counts::new();
    let mut out = Vec::with_capacity(plans.len());
    for (_, one) in ones {
        let one = one?;
        for (k, v) in one.counts {
            add(&mut counts, k, v);
        }
        out.push(one.plan);
    }
    std::fs::create_dir_all(stage(&o.out))?;
    write_jsonl(&stage(&o.out).join(PLAN5), &out)?;

    // ---- the validation slates of 4.3: v_tools' requests, v2's test rule with the fill of the
    // natural kind within the cell of the sources' texts, 20% removed, seed 1238
    let (main, masked) = v5_tools(&world, &o.v2, &kept, &pools)?;
    let val_dir = o.out.join("val");
    write_jsonl(&val_dir.join("v5_tools.jsonl"), &main)?;
    write_jsonl(&val_dir.join("v5_tools_masked.jsonl"), &masked)?;

    // ---- the false-negative job (4.2): every hard slate, an answer of v2 reused by its id
    let able_v2: HashSet<String> = read_or_empty::<Able>(&answers_path(&o.v2, "false-negatives"))?
        .into_iter()
        .map(|x| x.id)
        .collect();
    let mut items = Vec::new();
    let (mut reused, mut by_split) = (0usize, Counts::new());
    for p in &out {
        if p.split != "train" && p.split != "val" {
            bail!("{}: a false-negative item of the {} split; the teacher sees training and validation requests only", p.id, p.split);
        }
        if able_v2.contains(&p.fn_id) {
            reused += 1;
            continue;
        }
        bump(&mut by_split, &p.split);
        items.push(FnItem {
            id: p.fn_id.clone(),
            split: p.split.clone(),
            request: p.user.clone(),
            options: p
                .hard
                .tools
                .iter()
                .map(|id| l.tools[l.index[id]].option())
                .collect(),
        });
    }
    write_jsonl(&o.out.join(FN_STAGED), &items)?;
    // the group's training pools, for N2 (4.4)
    let tool_ids = |v: &[usize]| v.iter().map(|&t| l.tools[t].id.clone()).collect::<Vec<_>>();
    let intents_all: Vec<usize> = {
        let mut v: Vec<usize> = pools.intents.values().flatten().copied().collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    write_json(
        &o.out.join(POOLS5),
        &json!({
            "about": "s6-route v5: the training pool of each group (N2 of docs/s6-route5-prereg.md 4.4 reads the original descriptions of these tools); the intents' pool is every intent of the intents' training and validation requests, one pool per source in v2, joined",
            TOOLS: tool_ids(&pools.train), WHEN2CALL: tool_ids(&pools.train), INTENTS: tool_ids(&intents_all),
        }),
    )?;
    // ---- C6 at plan time (amendment 1): the planned slates as the build would write them
    let mut sel = Counts::new();
    let finals = planned_finals(&out, &mut sel);
    let chosen = choose(&out, &finals, o.seed, &mut sel);
    let mut cells = level_counts(&out, &finals, &chosen);
    cells.insert("val v5_tools.jsonl".into(), file_counts(&main));
    cells.insert("val v5_tools_masked.jsonl".into(), file_counts(&masked));
    let report = json!({
        "about": "s6-route v5 (docs/s6-route5-prereg.md, section 4, amendment 1): the plan, counted before the teacher",
        "seed": o.seed,
        "v5_tools seed": V5_VAL_SEED,
        "minimum": {"tools and when2call": MIN_CANDIDATES, "intents": format!("the slate's drawn distractors plus {INTENTS_SPARE}")},
        "cut points (the largest value of the classes 0 to 3)": {
            "F2 name length in characters": surface::Cuts::points(&cuts.names),
            "F3 description length in words": surface::Cuts::points(&cuts.words),
            "texts of the catalogue": cuts.names.len(),
            "tools of the catalogue": cat.len(),
        },
        "classes of F2 and F3": {"names": surface::Cuts::classes(&cuts.names),
                                 "words": surface::Cuts::classes(&cuts.words)},
        "plans": out.len(),
        "counts": counts,
        "the slates as planned, chosen as the build chooses them (no request dropped, no distractor named)": sel,
        "C6 bar": C6_BAR,
        "C6 cells judged from (drawn slates; amendment 2)": MIN_JUDGED,
        "C6 cells": c6_json(&cells),
        "v5_tools": {"slates": main.len(), "removed": main.iter().filter(|r| r.r.kind == "removed").count()},
        "teacher items": {"false-negatives": items.len(), "calls of 20": items.len().div_ceil(20),
                          "by split": by_split, "hard slates": out.len(),
                          "answered by v2's answer with the same id (not sent)": reused},
    });
    write_json(&o.out.join("counts/v5-plan.json"), &report)?;
    Ok(report)
}

/// `v5_tools` and `v5_tools_masked` (4.3): the requests of `v_tools` in order, one slate each by
/// v2's test rule, the fill of the natural kind within the cell of the sources' texts, 20%
/// removed, seed 1238.
fn v5_tools(
    w: &World,
    v2: &Path,
    kept: &[Kept],
    pools: &Pools,
) -> Result<(Vec<Record5>, Vec<Record5>)> {
    let l = w.l;
    let vcx = Ctx {
        l,
        seed: V5_VAL_SEED,
    };
    let reqs = build::tool_reqs(kept, "val");
    let v2_val: Vec<RouteRecord> = read_lines(&v2.join("val/v_tools.jsonl"))?;
    let want: Vec<&str> = v2_val.iter().map(|r| r.request.as_str()).collect();
    let got: Vec<&str> = reqs.iter().map(|k| k.r.id.as_str()).collect();
    if want != got {
        bail!("v5_tools would not hold the requests of v_tools, in order");
    }
    let pool = &pools.val;
    let removed = slates::removed_set(
        reqs.len(),
        REMOVED_SHARE,
        &mut vcx.rng(&format!("route/{V5_TOOLS}/removed")),
    );
    let kf = |t: usize| (w.keys.orig[t], w.keys.orig[t]);
    let df = |_: usize| false;
    let (mut main, mut masked) = (Vec::new(), Vec::new());
    for (i, k) in reqs.iter().enumerate() {
        let own: Vec<usize> = k
            .own
            .iter()
            .copied()
            .filter(|t| pool.binary_search(t).is_ok())
            .collect();
        let tok = v4::test_ok(&vcx, k);
        let own_t = own.iter().filter(|&&t| Some(t) != k.gold && tok(t)).count();
        let drawn = slates::TEST_TOOLS - 1 - own_t.min(slates::TEST_TOOLS - 1);
        let anchor = k.gold.map(|g| w.keys.orig[g]);
        let mut cell = Cell::new(
            pool,
            anchor,
            &w.table,
            &kf,
            &df,
            &tok,
            minimum(TOOLS, drawn),
        );
        let rem = removed.contains(&i);
        let lr = cell.level_for(minimum(TOOLS, drawn + 1));
        let src = w.source("tools/val", &k.r.source);
        let mut rng = vcx.rng(&format!("route/{V5_TOOLS}/slate/{}", k.r.id));
        let (s, level, drawn_f) = test_like(
            k.gold,
            &own,
            &mut cell,
            rem.then_some(lr),
            src,
            pool,
            &mut rng,
            &tok,
        );
        let rl = roles(&drawn_f, s.gold);
        let kind = if rem { "removed" } else { "test" };
        main.push(Record5 {
            r: vcx.record(V5_TOOLS, "val", k.r.id.clone(), kind, k, &s),
            cell_level: level,
            roles: rl.clone(),
            from: None,
        });
        let mut rng = vcx.rng(&format!("route/{V5_TOOLS}/masked/{}", k.r.id));
        let m = slates::masked(&s, |t| l.tools[t].desc.clone(), &mut rng);
        masked.push(Record5 {
            r: vcx.record(
                &format!("{V5_TOOLS}-masked"),
                "val",
                k.r.id.clone(),
                "masked",
                k,
                &m,
            ),
            cell_level: level,
            roles: rl,
            from: None,
        });
    }
    Ok((main, masked))
}

// -------------------------------------------------------------------------------- the slates

pub struct SlatesOptions {
    pub v2: PathBuf,
    pub v4: PathBuf,
    pub out: PathBuf,
    pub seed: u64,
    /// Without every false-negative answer: such a build is not trained on (smoke runs only).
    pub provisional: bool,
}

/// Step 4 of v5: the false-negative filter, the records (D1, D2, D4, D5, the levels and the
/// roles), the probe copies and the model's copies of the validation files.
#[allow(clippy::too_many_lines)]
pub fn slates(o: &SlatesOptions) -> Result<Value> {
    if o.out == o.v2 || o.out == o.v4 {
        bail!(
            "v5 writes to its own folder, never to v2's or v4's ({})",
            o.out.display()
        );
    }
    let tools: Vec<Tool> = read_lines(&stage(&o.v2).join("tools.jsonl"))?;
    let index: HashMap<String, usize> = tools
        .iter()
        .enumerate()
        .map(|(i, t)| (t.id.clone(), i))
        .collect();
    let plans: Vec<Plan5> = read_lines(&stage(&o.out).join(PLAN5))?;
    let rw = Rewrites::load(&o.v2, &o.v4)?;
    // the teacher's answers: v5's, then v2's for an item of the same id (4.1)
    let p5 = answers_path(&o.out, "false-negatives");
    let mut able: HashMap<String, Vec<usize>> = read_or_empty::<Able>(&p5)?
        .into_iter()
        .map(|x| (x.id, x.able))
        .collect();
    let refused = refused_ids(&p5)?;
    let fids: HashSet<&str> = plans.iter().map(|p| p.fn_id.as_str()).collect();
    let mut from_v2 = 0usize;
    for x in read_or_empty::<Able>(&answers_path(&o.v2, "false-negatives"))? {
        if fids.contains(x.id.as_str()) && !able.contains_key(&x.id) {
            from_v2 += 1;
            able.insert(x.id, x.able);
        }
    }
    let mut counts = Counts::new();
    let mut missing = 0usize;
    let mut finals: Vec<Option<Final>> = Vec::with_capacity(plans.len());
    for p in &plans {
        let verdict = match able.get(&p.fn_id) {
            Some(a) => Verdict::Able(a.clone()),
            None if refused.contains(&p.fn_id) => Verdict::Refused,
            None => {
                missing += 1;
                Verdict::Planned
            }
        };
        finals.push(
            filter_hard(p, &verdict, &mut counts).map(|(hard, hard_removed)| Final {
                natural: p.natural.clone(),
                natural_removed: p.natural_removed.clone(),
                hard,
                hard_removed,
                renamed: p.renamed.clone(),
                renamed_removed: p.renamed_removed.clone(),
            }),
        );
    }
    if missing > 0 && !o.provisional {
        bail!(
            "the teacher has not answered every false-negative item ({missing} missing): \
             BUILD={} scripts/route-teacher.sh false-negatives loop, or --provisional (not trained on)",
            o.out.display()
        );
    }
    let chosen = choose(&plans, &finals, o.seed, &mut counts);
    let cells = level_counts(&plans, &finals, &chosen);
    let mut shown: BTreeMap<String, Counts> = BTreeMap::new();
    let mut written = BTreeMap::new();
    let mut shares = BTreeMap::new();
    for ((group, split), picks) in &chosen {
        let c = shown.entry(format!("{group} {split}")).or_default();
        let mut all = Vec::with_capacity(picks.len());
        for pk in picks {
            let p = &plans[pk.req];
            let pl = pk.planned(&finals);
            let masked = (pk.kind == "renamed")
                .then(|| Rng::derived(o.seed, &format!("route5/renamed/{}", p.id)));
            all.push(record(
                o.seed,
                p,
                pl,
                pk.kind,
                pk.from(),
                &uid_of(p, pk),
                masked,
                &tools,
                &index,
                &rw,
                c,
            )?);
        }
        let is_none = |r: &Record5| r.r.target.last() == Some(&1.0);
        let mut kinds: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for r in &all {
            let e = kinds.entry(r.r.kind.clone()).or_default();
            e.0 += 1;
            e.1 += is_none(r) as usize;
        }
        let ren: Vec<&Record5> = all.iter().filter(|r| r.r.kind == "renamed").collect();
        let ren_none = ren.iter().filter(|r| is_none(r)).count();
        let other_n = all.len() - ren.len();
        let n_none = all.iter().filter(|r| is_none(r)).count();
        shares.insert(
            format!("{group} {split}"),
            json!({
                "slates": all.len(), "none gold": n_none, "share": n_none as f64 / all.len().max(1) as f64,
                "renamed": {"slates": ren.len(), "none gold": ren_none, "share": ren_none as f64 / ren.len().max(1) as f64},
                "not renamed": {"slates": other_n, "none gold": n_none - ren_none, "share": (n_none - ren_none) as f64 / other_n.max(1) as f64},
                "by kind (slates, none gold)": kinds,
            }),
        );
        let path = o
            .out
            .join("records")
            .join(group)
            .join(format!("{split}.jsonl"));
        written.insert(format!("{group}/{split}"), write_jsonl(&path, &all)?);
    }

    // ---- the probe copies of the validation slates (4.6), and the model's copies (D4)
    let val = o.out.join("val");
    let main: Vec<Record5> = read_lines(&val.join("v5_tools.jsonl"))?;
    let masked: Vec<Record5> = read_lines(&val.join("v5_tools_masked.jsonl"))?;
    if main.len() != masked.len()
        || main
            .iter()
            .zip(&masked)
            .any(|(a, b)| a.r.request != b.r.request)
    {
        bail!("v5_tools and v5_tools_masked do not hold the same slates");
    }
    let (mut main_rw, mut masked_rw) = (Vec::new(), Vec::new());
    let mut probe = Counts::new();
    for (r, m) in main.iter().zip(&masked) {
        let g =
            r.r.target
                .iter()
                .position(|&x| x == 1.0)
                .unwrap_or(r.r.tools.len());
        let cand: Vec<usize> = (0..r.r.tools.len())
            .filter(|&i| i != g && rw.text.contains_key(&r.r.tools[i]))
            .collect();
        if cand.is_empty() {
            bump(
                &mut probe,
                "slates without a rewritten distractor (left out)",
            );
            continue;
        }
        let mut rng = Rng::derived(V5_VAL_SEED, &format!("route/{V5_TOOLS}-rw/{}", r.r.request));
        let d = cand[rng.below(cand.len())];
        let desc = &rw.text[&r.r.tools[d]];
        for (src, dst) in [(r, &mut main_rw), (m, &mut masked_rw)] {
            let mut x = src.clone();
            let name = sky_infer::route::option_name(&src.r.opts[d]).to_string();
            x.r.opts[d] = option_text(&name, desc);
            x.r.follow = Some(d);
            x.r.src = format!("{}-rw", src.r.src);
            dst.push(x);
        }
        bump(
            &mut probe,
            if g < r.r.tools.len() {
                "tool-gold slates"
            } else {
                "none-gold slates"
            },
        );
    }
    write_jsonl(&val.join("v5_tools_rw.jsonl"), &main_rw)?;
    write_jsonl(&val.join("v5_tools_masked_rw.jsonl"), &masked_rw)?;
    for f in [
        "v5_tools",
        "v5_tools_masked",
        "v5_tools_rw",
        "v5_tools_masked_rw",
    ] {
        build::render_with(
            &val.join(format!("{f}.jsonl")),
            &val.join(format!("{f}.model.jsonl")),
            Render::ScoresInOptions,
        )?;
    }
    // the reported readings of the pilot (5): v4's v4_tools and v2's v_tools, D4 copies
    for (dir, f) in [
        (&o.v4, "v4_tools"),
        (&o.v4, "v4_tools_masked"),
        (&o.v2, "v_tools"),
        (&o.v2, "v_tools_masked"),
    ] {
        let p = dir.join("val").join(format!("{f}.jsonl"));
        if p.exists() {
            build::render_with(
                &p,
                &val.join(format!("{f}.model.jsonl")),
                Render::ScoresInOptions,
            )?;
        } else {
            bump(&mut probe, format!("{} absent: no model copy", p.display()));
        }
    }
    // the renamed and natural validation records of the tools group (4.6)
    if let Ok(recs) = read_lines::<Record5>(&o.out.join("records/tools/val.jsonl")) {
        for kind in ["renamed", "natural"] {
            let v: Vec<&Record5> = recs.iter().filter(|r| r.r.kind == kind).collect();
            write_jsonl(&val.join(format!("r_tools_{kind}.model.jsonl")), &v)?;
        }
    }
    let mut all_cells = cells;
    all_cells.insert("val v5_tools.jsonl".into(), file_counts(&main));
    all_cells.insert("val v5_tools_masked.jsonl".into(), file_counts(&masked));
    let dropped = plans.len() - finals.iter().filter(|f| f.is_some()).count();
    let report = json!({
        "about": "s6-route v5 (docs/s6-route5-prereg.md, section 4): the records, counted before any training",
        "seed": o.seed,
        "provisional": o.provisional,
        "false-negative answers missing": missing,
        "teacher (4.2, 9)": {"hard slates": plans.len(), "answered by v2's answer with the same id": from_v2,
                             "refused by moderation": plans.iter().filter(|p| refused.contains(&p.fn_id)).count(),
                             "requests dropped": dropped},
        "records": written,
        "none share": shares,
        "rewrites shown (D1), by group and split": shown,
        "probe copies (4.6)": probe,
        "C6 bar": C6_BAR,
        "C6 cells": c6_json(&all_cells),
        "counts": counts,
    });
    write_json(&o.out.join("counts/v5-slates.json"), &report)?;
    Ok(report)
}

/// One record: names real or `tool_<k>` (the renamed kind's stream), D1 on every tool option,
/// D4's rendering, the slate's level and the options' roles.
#[allow(clippy::too_many_arguments)]
fn record(
    seed: u64,
    p: &Plan5,
    pl: &Planned,
    kind: &str,
    from: Option<&str>,
    uid: &str,
    masked: Option<Rng>,
    tools: &[Tool],
    index: &HashMap<String, usize>,
    rw: &Rewrites,
    c: &mut Counts,
) -> Result<Record5> {
    let ts = &pl.tools;
    let gold = p.gold.as_ref().and_then(|g| ts.iter().position(|t| t == g));
    let ks: Option<Vec<usize>> = masked.map(|mut rng| {
        let mut ks: Vec<usize> = (1..=99).collect();
        rng.shuffle(&mut ks);
        ks
    });
    let mut opts = Vec::with_capacity(ts.len() + 1);
    let mut flags = Vec::with_capacity(ts.len());
    for (i, id) in ts.iter().enumerate() {
        let t = &tools[*index
            .get(id)
            .with_context(|| format!("{}: unknown tool {id}", p.id))?];
        let mut desc = t.desc.clone();
        let mut flag = false;
        bump(c, "tool options");
        // D1 by the request and the kind of the slate whose text the record shows: a removed
        // copy shows its source slate's (amendment 2, decision 1)
        if v4::rewrite_draw(seed, &p.id, from.unwrap_or(kind), id) {
            bump(c, "drawn for the rewrite");
            if let Some(d) = rw.text.get(id) {
                desc = d.clone();
                flag = true;
            } else if rw.refused.contains(id) {
                bump(c, "drawn, the rewrite refused: shown original");
            } else {
                bump(c, "drawn, no rewrite: shown original");
            }
        }
        let name = match &ks {
            Some(ks) => format!("tool_{}", ks[i]),
            None => t.name.clone(),
        };
        opts.push(option_text(&name, &desc));
        flags.push(flag);
    }
    let tool_opts: Vec<&str> = opts.iter().map(String::as_str).collect();
    let mut model_opts = options_with_scores(&p.user, &tool_opts);
    model_opts.push(NONE_OPTION.to_string());
    let mut target = vec![0.0; model_opts.len()];
    target[gold.unwrap_or(model_opts.len() - 1)] = 1.0;
    Ok(Record5 {
        r: RouteRecord {
            src: format!("{}:{}", p.group, p.source),
            t: "choice".into(),
            ins: INSTRUCTION.into(),
            opts: model_opts,
            state: super::state_of(&p.user),
            target,
            split: p.split.clone(),
            uid: uid.into(),
            kind: kind.into(),
            lang: p.lang.clone(),
            request: p.id.clone(),
            tools: ts.to_vec(),
            follow: None,
            rewritten: Some(flags),
        },
        cell_level: pl.level,
        roles: roles(&pl.drawn, gold),
        from: from.map(str::to_string),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route::build::items_path;
    use crate::route::build::tests::{fixture, hashes, run, write_npy};

    /// The real pools (`SKY_ROUTE_V2=<v2 build> SKY_ROUTE_V4=<v4 build>`; read only, nothing
    /// written, no request embedded): v2's split recomputed, every tool of the pools with a
    /// rewrite in v2's or v4's answers, the cut points and the keys computed, `tool_<k>` in one
    /// length class, v2's plan of training and validation requests only. Counts only.
    /// `cargo test -p sky-data --lib real_pools -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn real_pools_are_read() {
        let (Ok(v2), Ok(v4d)) = (std::env::var("SKY_ROUTE_V2"), std::env::var("SKY_ROUTE_V4"))
        else {
            return;
        };
        let (v2, v4d) = (PathBuf::from(v2), PathBuf::from(v4d));
        let V2 { l, a, kept } = v4::load_v2(&v2, v4::V2_SEED).unwrap();
        let pools = Pools::new(&a, &kept);
        let cat = catalogue(&pools);
        let rw = rewrites(&v2, &v4d, &cat, &l).unwrap();
        let (cuts, keys) = keys_of(&l, &rw, &cat).unwrap();
        let plans: Vec<Plan> = read_lines(&stage(&v2).join("plan.jsonl")).unwrap();
        assert!(plans.iter().all(|p| p.split == "train" || p.split == "val"));
        // amendment 2, reading 7: the keys of every text, for scripts/s6-route5-checks.py
        // --compare-keys (SKY_ROUTE_V5_KEYS: a file outside the data folder)
        if let Ok(path) = std::env::var("SKY_ROUTE_V5_KEYS") {
            let mut rows = vec![json!({
                "classes of F2 and F3": {"names": surface::Cuts::classes(&cuts.names),
                                         "words": surface::Cuts::classes(&cuts.words)},
                "styles": surface::STYLES,
            })];
            for &t in &cat {
                let tool = &l.tools[t];
                let mut texts = vec![tool.desc.clone()];
                texts.extend(rw.text.get(&tool.id).cloned());
                for d in texts {
                    let k = cuts.key(&tool.name, &d);
                    rows.push(json!({"id": tool.id, "option": option_text(&tool.name, &d),
                                     "key": [k.style, k.name_q, k.words_q, k.period as u8, k.form, k.schema]}));
                }
            }
            write_jsonl(Path::new(&path), &rows).unwrap();
        }
        let distinct =
            |pool: &[usize], k: &[u16]| pool.iter().map(|&t| k[t]).collect::<HashSet<_>>().len();
        let intents: Vec<usize> = pools.intents.values().flatten().copied().collect();
        println!(
            "{}",
            json!({
                "tools of the catalogue": cat.len(), "texts": cuts.names.len(),
                "refused rewrites in the catalogue": cat.iter().filter(|&&t| !rw.text.contains_key(&l.tools[t].id)).count(),
                "F2 cut points": surface::Cuts::points(&cuts.names),
                "F3 cut points": surface::Cuts::points(&cuts.words),
                "pools": {"train": pools.train.len(), "val": pools.val.len(), "w2c val": pools.w2c_val.len(),
                          "intents": pools.intents.iter().map(|(k, v)| (k.clone(), v.len())).collect::<BTreeMap<_, _>>()},
                "non-empty cells (originals, rewrites)": {
                    "train": [distinct(&pools.train, &keys.orig), distinct(&pools.train, &keys.rw)],
                    "val": [distinct(&pools.val, &keys.orig), distinct(&pools.val, &keys.rw)],
                    "intents": [distinct(&intents, &keys.orig), distinct(&intents, &keys.rw)],
                },
                "plans": plans.len(),
            })
        );
    }

    #[test]
    fn d1_is_v4s_draw() {
        for (r, k, t) in [
            ("glaive:6", "natural", "t79e528816000f2ac"),
            ("when2call:n3", "hard", "t0123456789abcdef"),
            ("massive:12", "renamed", "tffffffffffffffff"),
        ] {
            let d = D1::new(1234, r, k);
            for i in 0..200 {
                let tool = format!("{t}{i}");
                assert_eq!(
                    d.draw(&tool),
                    v4::rewrite_draw(1234, r, k, &tool),
                    "{r} {k} {tool}"
                );
            }
        }
    }

    #[test]
    fn the_six_features() {
        use surface::*;
        assert_eq!(schema_words("Gets the data of a given list."), 3);
        assert_eq!(schema_words("Gets metadata, listing and IDs."), 1);
        assert_eq!(schema_words("Based on the API parameters: JSON."), 4);
        assert_eq!(schema_words("parameter_x and parameters"), 1);
        assert_eq!(schema_words("based  on"), 0);
        assert_eq!(first_form("Gets the weather."), 0);
        assert_eq!(first_form("Address book."), 2);
        assert_eq!(first_form("Get the weather."), 1);
        assert_eq!(first_form("get the weather."), 2);
        assert_eq!(first_form("Get-weather now"), 2);
        assert_eq!(first_form(""), 2);
        for i in [0, 1, 17, 777, KEYS - 1] {
            assert_eq!(Key::from_index(i).index(), i);
        }
        let cuts = Cuts::new([
            ("abc", "one two"),
            ("abcdefgh", "one two three four five six"),
        ]);
        let k = cuts.key("get_x", "Gets the data.");
        assert_eq!((k.style, k.period, k.form, k.schema), (0, true, 0, 1));
        // a value equal to a cut point goes to the lower class
        let sorted: Vec<usize> = (1..=10).collect();
        assert_eq!(quintile(&sorted, 2), 0);
        assert_eq!(quintile(&sorted, 3), 1);
        assert_eq!(
            Cuts::points(&sorted),
            vec![Some(2), Some(4), Some(6), Some(8)]
        );
    }

    #[test]
    fn the_levels_are_nested_and_widen_in_order() {
        use surface::*;
        let a = Key {
            style: 0,
            name_q: 2,
            words_q: 2,
            period: true,
            form: 1,
            schema: 0,
        };
        let cases = [
            (a, 0),
            (
                Key {
                    name_q: 3,
                    words_q: 1,
                    ..a
                },
                1,
            ),
            (Key { schema: 2, ..a }, 2),
            (
                Key {
                    form: 0,
                    schema: 2,
                    ..a
                },
                3,
            ),
            (
                Key {
                    style: 3,
                    form: 0,
                    ..a
                },
                4,
            ),
            (
                Key {
                    style: 3,
                    name_q: 0,
                    ..a
                },
                5,
            ),
            (Key { words_q: 4, ..a }, 6),
            (Key { period: false, ..a }, 6),
        ];
        for (b, want) in cases {
            assert_eq!(match_level(a, b), want, "{b:?}");
            for l in 0..=L6 {
                assert_eq!(matches(a, b, l), l >= want, "{b:?} at {l}");
            }
        }
        let t = level_table();
        assert_eq!(t[a.index() * KEYS + cases[4].0.index()], 4);
    }

    fn never(_: usize) -> bool {
        false
    }

    #[test]
    fn a_slate_is_built_at_the_lowest_level_that_holds_its_minimum() {
        use surface::*;
        let table = level_table();
        let a = Key {
            style: 0,
            name_q: 2,
            words_q: 2,
            period: true,
            form: 1,
            schema: 0,
        };
        // 5 tools at L0, 6 more at L1, 20 at L3, 40 at L6; tool 0 refused by the filter
        let mut ks: Vec<Key> = vec![a; 5];
        ks.extend(vec![Key { name_q: 3, ..a }; 6]);
        ks.extend(vec![
            Key {
                form: 2,
                schema: 1,
                ..a
            };
            20
        ]);
        ks.extend(vec![Key { period: false, ..a }; 40]);
        let idx: Vec<u16> = ks.iter().map(|k| k.index() as u16).collect();
        let keys = move |t: usize| (idx[t], idx[t]);
        let ok = |t: usize| t != 0;
        let pool: Vec<usize> = (0..ks.len()).collect();
        let anchor = Some(a.index() as u16);
        let cell = |min| super::Cell::new(&pool, anchor, &table, &keys, &never, &ok, min);
        // tools and When2Call: 16; 4 + 6 = 10 by L1, 30 by L3
        let mut c = cell(minimum(TOOLS, 6));
        assert_eq!(c.level, 3);
        assert_eq!(c.count(1), 10);
        assert_eq!(c.members(0), vec![1, 2, 3, 4]);
        assert_eq!(c.members(3).len(), 30);
        assert_eq!(c.level_for(31), L6);
        // amendment 1: an intents slate's minimum is its drawn distractors plus 3
        assert_eq!(minimum(INTENTS, 1), 4);
        assert_eq!(minimum(INTENTS, 7), 10);
        assert_eq!(minimum(WHEN2CALL, 1), MIN_CANDIDATES);
        assert_eq!(cell(minimum(INTENTS, 1)).level, 0);
        assert_eq!(cell(minimum(INTENTS, 6)).level, 1);
        assert_eq!(cell(minimum(INTENTS, 7)).level, 1);
        assert_eq!(cell(minimum(INTENTS, 8)).level, 3);
        // the fill draws within the level first, then from the next level once it is used up
        let mut c = cell(minimum(INTENTS, 1));
        let mut s = vec![0usize];
        let mut rng = Rng::new(3);
        let lv = fill_levels(&mut s, 7, 0, &mut c, &[], &pool, &mut rng, &ok);
        assert_eq!(s.len(), 7);
        assert_eq!(lv, vec![0, 0, 0, 0, 1, 1]);
        assert!(s[1..5].iter().all(|&t| t < 5) && s[5..].iter().all(|&t| (5..11).contains(&t)));
        // the source's tools first, within the level (D7)
        let mut c = cell(minimum(INTENTS, 1));
        let mut s = vec![0usize];
        fill_levels(&mut s, 3, 3, &mut c, &[20, 21, 60], &pool, &mut rng, &ok);
        assert_eq!(
            s[1..].iter().copied().collect::<HashSet<_>>(),
            HashSet::from([20, 21])
        );
        // no anchor: outside the cell
        assert_eq!(
            super::Cell::new(&pool, None, &table, &keys, &never, &ok, 4).level,
            L6
        );
    }

    #[test]
    fn the_cell_reads_the_text_d1_shows() {
        use surface::*;
        let table = level_table();
        let a = Key {
            style: 0,
            name_q: 2,
            words_q: 2,
            period: true,
            form: 1,
            schema: 0,
        };
        let b = Key { period: false, ..a };
        // every tool: the original in the cell, the rewrite outside; D1 shows the rewrite of the
        // odd tools
        let keys = |_: usize| (a.index() as u16, b.index() as u16);
        let draw = |t: usize| t % 2 == 1;
        let ok = |_: usize| true;
        let pool: Vec<usize> = (0..40).collect();
        let mut c = super::Cell::new(&pool, Some(a.index() as u16), &table, &keys, &draw, &ok, 16);
        assert_eq!(c.level, 0);
        assert_eq!(c.members(0), (0..40).step_by(2).collect::<Vec<_>>());
        assert_eq!((c.level_of(2), c.level_of(3)), (0, L6));
    }

    fn planned(tools: &[&str], level: u8) -> Planned {
        Planned {
            tools: tools.iter().map(|s| s.to_string()).collect(),
            drawn: tools.iter().map(|&t| t != "g").collect(),
            level,
        }
    }

    fn plan5(hard: Planned, more: &[(&str, u8)]) -> Plan5 {
        Plan5 {
            id: "glaive:1".into(),
            group: TOOLS.into(),
            source: "glaive".into(),
            split: "train".into(),
            lang: "en".into(),
            user: "do it".into(),
            kind: "call".into(),
            gold: Some("g".into()),
            natural: planned(&["g"], 0),
            natural_removed: None,
            hard,
            hard_removed_min: 1,
            hard_more: more.iter().map(|(t, _)| t.to_string()).collect(),
            hard_more_levels: more.iter().map(|(_, l)| *l).collect(),
            renamed: None,
            renamed_removed: None,
            fn_id: "glaive:1#x".into(),
        }
    }

    fn strs(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn a_replacement_from_the_next_level_raises_the_slates_level() {
        let p = plan5(
            planned(&["a", "g", "b"], 1),
            &[("c", 1), ("d", 3), ("e", 3)],
        );
        let mut c = Counts::new();
        // the teacher names a and b: a takes c (L1), b takes d (L3); the slate is raised to L3,
        // D5's extra is e (L3)
        let (h, r) = filter_hard(&p, &Verdict::Able(vec![1, 2, 3]), &mut c).unwrap();
        assert_eq!((h.tools, h.level), (strs(&["c", "g", "d"]), 3));
        let r = r.unwrap();
        assert_eq!(r.tools, strs(&["c", "e", "d"]));
        assert_eq!((r.level, r.drawn), (3, vec![true, true, true]));
        // the teacher names a: c takes its place at the slate's level; the removed copy takes d,
        // from L3, and is raised alone
        let (h, r) = filter_hard(&p, &Verdict::Able(vec![1, 2]), &mut c).unwrap();
        assert_eq!((h.tools, h.level), (strs(&["c", "g", "b"]), 1));
        assert_eq!(r.unwrap().level, 3);
        // the plan before the teacher: the removed copy at its own minimum's level
        let (h, r) = filter_hard(&p, &Verdict::Planned, &mut c).unwrap();
        assert_eq!((h, r.unwrap().level), (p.hard.clone(), 1));
        // the teacher does not name the gold: the request is dropped
        assert!(filter_hard(&p, &Verdict::Able(vec![1]), &mut c).is_none());
        assert_eq!(
            c["tools train: dropped, the teacher says the gold cannot"],
            1
        );
        // moderation refused: unfiltered, counted
        let (h, _) = filter_hard(&p, &Verdict::Refused, &mut c).unwrap();
        assert_eq!(h, p.hard);
        assert_eq!(
            c["tools train: refused by moderation, hard slate unfiltered"],
            1
        );
        // a named tool never comes back; the sequence used up is counted, and the removed copy
        // is then the slate without its gold (as v4)
        let p = plan5(planned(&["a", "g", "b"], 0), &[("a", 0), ("c", 2)]);
        let (h, r) = filter_hard(&p, &Verdict::Able(vec![1, 2, 3]), &mut c).unwrap();
        assert_eq!((h.tools, h.level), (strs(&["c", "g", "b"]), 2));
        assert_eq!(r.unwrap().tools, strs(&["c", "b"]));
        assert_eq!(c["tools train: no tool left to replace a distractor"], 1);
    }

    /// The rewrites of every tool of the fixture (as v4's teacher answered them), one refused.
    fn v4_rewrites(v2: &Path, v4: &Path) {
        let tools: Vec<Tool> = read_lines(&stage(v2).join("tools.jsonl")).unwrap();
        let ans: Vec<Value> = tools[1..]
            .iter()
            .enumerate()
            .map(|(j, t)| {
                let d = if j % 3 == 0 {
                    format!("Returns the api data of {} for a given list of ids", t.name)
                } else {
                    format!("Get {} now.", t.name)
                };
                json!({"id": t.id, "description": d})
            })
            .collect();
        write_jsonl(&answers_path(v4, "rewrite-description"), &ans).unwrap();
        write_jsonl(
            &crate::distill::refused_path(&answers_path(v4, "rewrite-description")),
            &[json!({"id": tools[0].id, "src": "rewrite-description", "reason": "moderation"})],
        )
        .unwrap();
    }

    /// The request embeddings: each request with a gold of the tools group sits on a tool far
    /// from its gold (the thing 100 places away), the others on random rows.
    fn request_embeddings(v2: &Path, out: &Path) {
        let plans: Vec<Plan> = read_lines(&stage(v2).join("plan.jsonl")).unwrap();
        let tools: Vec<Tool> = read_lines(&stage(v2).join("tools.jsonl")).unwrap();
        let texts: Vec<Value> = read_lines(&stage(v2).join("texts.jsonl")).unwrap();
        let pot = Matrix::read_npy(&v2.join("emb/potion.npy")).unwrap();
        let row: HashMap<&str, usize> = texts
            .iter()
            .enumerate()
            .map(|(i, t)| (t["id"].as_str().unwrap(), i))
            .collect();
        let name_of: HashMap<&str, &str> = tools
            .iter()
            .map(|t| (t.id.as_str(), t.name.as_str()))
            .collect();
        let by_name: HashMap<&str, &str> = tools
            .iter()
            .map(|t| (t.name.as_str(), t.id.as_str()))
            .collect();
        let mut rng = Rng::new(9);
        let rows: Vec<Vec<f32>> = plans
            .iter()
            .map(|p| {
                let far = p.gold.as_ref().and_then(|g| {
                    let n = name_of[g.as_str()];
                    let i: usize = n
                        .strip_prefix("get_thing")?
                        .split('_')
                        .next()?
                        .parse()
                        .ok()?;
                    by_name
                        .get(format!("get_thing{}_v0", (i + 100) % 200).as_str())
                        .copied()
                });
                match far {
                    Some(f) => pot.row(row[f]).to_vec(),
                    None => (0..pot.dim).map(|_| rng.f64() as f32 - 0.5).collect(),
                }
            })
            .collect();
        std::fs::create_dir_all(out.join("emb")).unwrap();
        write_npy(&out.join(REQ_EMB), &rows);
        write_json(
            &out.join(REQ_META),
            &json!({"source_sha256": sha256_file(&stage(v2).join("plan.jsonl")).unwrap(),
                    "ids_sha256": plan_ids_sha256(&plans), "rows": rows.len()}),
        )
        .unwrap();
    }

    /// The false-negative answers: the gold able, except (unless `clean`) some that name a
    /// distractor too and some that say the gold cannot.
    fn fn_answers(out: &Path, clean: bool) {
        let plans: Vec<Plan5> = read_lines(&stage(out).join(PLAN5)).unwrap();
        let items: Vec<FnItem> = read_lines(&out.join(FN_STAGED)).unwrap();
        let want: HashSet<&str> = items.iter().map(|i| i.id.as_str()).collect();
        let mut able = Vec::new();
        for (j, p) in plans.iter().enumerate() {
            if !want.contains(p.fn_id.as_str()) {
                continue;
            }
            let gp = p
                .gold
                .as_ref()
                .and_then(|g| p.hard.tools.iter().position(|t| t == g));
            let mut a: Vec<usize> = gp.map(|g| g + 1).into_iter().collect();
            if !clean {
                if j % 17 == 3 {
                    a.clear();
                } else if j % 5 == 1 {
                    if let Some(i) = (0..p.hard.tools.len()).find(|&i| Some(i) != gp) {
                        a.push(i + 1);
                    }
                }
            }
            a.sort_unstable();
            able.push(json!({"id": p.fn_id, "able": a}));
        }
        write_jsonl(&answers_path(out, "false-negatives"), &able).unwrap();
    }

    fn opts(v2: &Path, v4: &Path, out: &Path) -> PlanOptions {
        PlanOptions {
            v2: v2.into(),
            v4: v4.into(),
            out: out.into(),
            seed: 7,
        }
    }

    fn v5_run(v2: &Path, v4: &Path, out: &Path) -> Value {
        request_embeddings(v2, out);
        plan(&opts(v2, v4, out)).unwrap()
    }

    fn v5_slates(v2: &Path, v4: &Path, out: &Path, provisional: bool) -> Result<Value> {
        slates(&SlatesOptions {
            v2: v2.into(),
            v4: v4.into(),
            out: out.into(),
            seed: 7,
            provisional,
        })
    }

    fn stage_items(out: &Path) {
        std::fs::create_dir_all(out.join("teacher")).unwrap();
        std::fs::copy(out.join(FN_STAGED), items_path(out, "false-negatives")).unwrap();
    }

    /// The invented build of [`an_invented_v5_build`], kept in `SKY_ROUTE_V5_OUT` (a folder
    /// outside the data folder), for `scripts/s6-route5-checks.py` to read: `cargo test -p
    /// sky-data --lib invented_build_for -- --ignored`.
    #[test]
    #[ignore]
    fn invented_build_for_the_checks() {
        let Ok(root) = std::env::var("SKY_ROUTE_V5_OUT") else {
            return;
        };
        let root = PathBuf::from(root);
        let (v2, v4d, b5) = (root.join("v2"), root.join("v4"), root.join("b5"));
        fixture(&v2);
        run(&v2, 7);
        v4_rewrites(&v2, &v4d);
        v5_run(&v2, &v4d, &b5);
        stage_items(&b5);
        fn_answers(&b5, false);
        v5_slates(&v2, &v4d, &b5, false).unwrap();
    }

    #[test]
    fn an_invented_v5_build() {
        let root = std::env::temp_dir().join(format!("sky-route-v5-{}", std::process::id()));
        let (v2, v4d) = (root.join("v2"), root.join("v4"));
        fixture(&v2);
        run(&v2, 7);
        v4_rewrites(&v2, &v4d);
        // v2's false-negative answers: every one of v2's items, the gold able
        let v2_items: Vec<FnItem> = read_lines(&items_path(&v2, "false-negatives")).unwrap();
        let v2_plans: Vec<Plan> = read_lines(&stage(&v2).join("plan.jsonl")).unwrap();
        let v2_able: Vec<Value> = v2_items
            .iter()
            .zip(&v2_plans)
            .map(|(it, p)| {
                let g = p
                    .gold
                    .as_ref()
                    .and_then(|g| p.hard.iter().position(|t| t == g));
                json!({"id": it.id, "able": g.map(|g| vec![g + 1]).unwrap_or_default()})
            })
            .collect();
        write_jsonl(&answers_path(&v2, "false-negatives"), &v2_able).unwrap();
        let before = hashes(&v2);
        let (a, b, c) = (root.join("a"), root.join("b"), root.join("c"));
        let rep = v5_run(&v2, &v4d, &a);
        v5_run(&v2, &v4d, &b);
        // determinism under the seed
        assert_eq!(hashes(&a), hashes(&b));
        // without the teacher's answers, only a provisional build
        assert!(v5_slates(&v2, &v4d, &a, false).is_err());
        // the false-negative items: training and validation requests only, v2's item form, the
        // ones v2 answered with the same id (the one-tool hard slates) not sent, accepted by the
        // teacher's guard, which refuses a test item
        let items: Vec<Value> = read_lines(&a.join(FN_STAGED)).unwrap();
        let reused = rep["teacher items"]["answered by v2's answer with the same id (not sent)"]
            .as_u64()
            .unwrap();
        assert!(!items.is_empty() && reused > 0, "{rep}");
        let plans: Vec<Plan5> = read_lines(&stage(&a).join(PLAN5)).unwrap();
        assert_eq!(items.len() as u64 + reused, plans.len() as u64);
        for it in &items {
            let keys: HashSet<&str> = it.as_object().unwrap().keys().map(String::as_str).collect();
            assert_eq!(keys, HashSet::from(["id", "split", "request", "options"]));
            assert!(matches!(it["split"].as_str(), Some("train" | "val")));
        }
        crate::teacher::route_todo(&a.join(FN_STAGED), &answers_path(&a, "false-negatives"))
            .unwrap();
        let mut bad = items.clone();
        bad[0]["split"] = json!("test");
        let bp = root.join("bad.items.jsonl");
        write_jsonl(&bp, &bad).unwrap();
        assert!(crate::teacher::route_todo(&bp, &root.join("bad.jsonl")).is_err());
        stage_items(&a);
        fn_answers(&a, false);
        let srep = v5_slates(&v2, &v4d, &a, false).unwrap();
        // v2's folder is never written
        assert_eq!(hashes(&v2), before);

        // the tables of the checks below
        let l = build::load(&v2).unwrap();
        let asg = build::assign(&l).unwrap();
        let kept = asg.kept();
        let pools = Pools::new(&asg, &kept);
        let cat = catalogue(&pools);
        let rw = rewrites(&v2, &v4d, &cat, &l).unwrap();
        let (cuts, _) = keys_of(&l, &rw, &cat).unwrap();
        let by_req: HashMap<&str, &Kept> = kept.iter().map(|k| (k.r.id.as_str(), k)).collect();
        let reqm = Matrix::read_npy(&a.join(REQ_EMB)).unwrap();

        // S1′: the hard distractors follow the request's nearest tools, not the gold's (each
        // request of a tools gold sits on a tool far from its gold); v2's followed the gold
        let pot = &l.potion;
        let (mut req5, mut gold5, mut req2, mut gold2, mut n) = (0f32, 0f32, 0f32, 0f32, 0usize);
        for (i, (p, p2)) in plans.iter().zip(&v2_plans).enumerate() {
            let (Some(g), true) = (&p.gold, p.group == TOOLS) else {
                continue;
            };
            let gr = pot.row(l.row[l.index[g]].unwrap());
            let q = reqm.row(i);
            for (t, t2) in p.hard.tools.iter().zip(&p2.hard) {
                if t == g || t2 == g {
                    continue;
                }
                let (r, r2) = (
                    pot.row(l.row[l.index[t]].unwrap()),
                    pot.row(l.row[l.index[t2]].unwrap()),
                );
                req5 += cluster::dot(q, r);
                gold5 += cluster::dot(gr, r);
                req2 += cluster::dot(q, r2);
                gold2 += cluster::dot(gr, r2);
                n += 1;
            }
        }
        assert!(n > 100);
        let m = |x: f32| x / n as f32;
        assert!(
            m(req5) > m(gold5) + 0.2,
            "v5: {} to the request, {} to the gold",
            m(req5),
            m(gold5)
        );
        assert!(m(gold2) > m(req2) + 0.2, "v2: {} {}", m(gold2), m(req2));

        let tools: Vec<Tool> = read_lines(&stage(&v2).join("tools.jsonl")).unwrap();
        let by_id: HashMap<&str, &Tool> = tools.iter().map(|t| (t.id.as_str(), t)).collect();
        let key_of = |o: &str| {
            let plain = sky_infer::route::plain_option(o);
            let (nm, d) = plain.split_once(": ").unwrap_or((plain, ""));
            cuts.key(nm, d)
        };
        let mut all: Vec<Record5> = Vec::new();
        for g in [TOOLS, WHEN2CALL, INTENTS] {
            for s in ["train", "val"] {
                let recs: Vec<Record5> =
                    read_lines(&a.join(format!("records/{g}/{s}.jsonl"))).unwrap();
                assert!(!recs.is_empty(), "{g} {s}");
                // D2: none on 20% of the renamed slates of each group and split
                let ren: Vec<&Record5> = recs.iter().filter(|r| r.r.kind == "renamed").collect();
                let none = ren
                    .iter()
                    .filter(|r| r.r.target.last() == Some(&1.0))
                    .count();
                assert_eq!(none, (ren.len() as f64 * 0.2).round() as usize, "{g} {s}");
                all.extend(recs);
            }
        }
        let (mut l0, mut exempt, mut checked, mut widened) = (0, 0, 0, 0);
        for r in &all {
            assert!(r.cell_level <= surface::L6);
            assert_eq!(r.roles.len(), r.r.tools.len());
            let g = r.r.target.iter().position(|&x| x == 1.0).unwrap();
            let gold = (g < r.r.tools.len()).then_some(g);
            let k = by_req[r.r.request.as_str()];
            let own: HashSet<&str> = k.own.iter().map(|&t| l.tools[t].id.as_str()).collect();
            for (i, (role, id)) in r.roles.iter().zip(&r.r.tools).enumerate() {
                assert_eq!(role == RIGHT, Some(i) == gold, "{}", r.r.uid);
                if role == EXEMPT {
                    // the request's own tools, shown whatever their cell; in a hard slate only
                    // without a tool gold (amendment 2, decision 3)
                    assert!(own.contains(id.as_str()), "{}", r.r.uid);
                    assert!(
                        !r.r.uid.ends_with("/hard") || k.gold.is_none(),
                        "{}",
                        r.r.uid
                    );
                    exempt += 1;
                }
            }
            // D1 is v4's draw: the rewrite where the draw says so and the tool has one; a
            // removed copy draws as the slate it comes from (amendment 2, decision 1)
            assert_eq!(r.from.is_some(), r.r.kind == "removed", "{}", r.r.uid);
            let d1_kind = r.from.as_deref().unwrap_or(&r.r.kind);
            for (i, id) in r.r.tools.iter().enumerate() {
                let want =
                    v4::rewrite_draw(7, &r.r.request, d1_kind, id) && rw.text.contains_key(id);
                assert_eq!(r.r.rewritten.as_ref().unwrap()[i], want, "{}", r.r.uid);
            }
            // a slate below L6: every drawn option matches the right tool's key, as shown, at
            // the slate's level (at L0: shares it)
            let Some(g) = gold.filter(|_| r.cell_level < surface::L6) else {
                continue;
            };
            let kg = key_of(&r.r.opts[g]);
            for (i, role) in r.roles.iter().enumerate() {
                if role != DRAWN {
                    continue;
                }
                let ki = key_of(&r.r.opts[i]);
                assert!(
                    surface::matches(kg, ki, r.cell_level),
                    "{} option {i}",
                    r.r.uid
                );
                if r.cell_level == 0 {
                    assert_eq!(ki, kg);
                    checked += 1;
                }
            }
            l0 += (r.cell_level == 0) as usize;
            widened += (r.cell_level > 0) as usize;
        }
        assert!(
            l0 > 100 && checked > 100 && exempt > 50 && widened > 0,
            "{l0} {checked} {exempt} {widened}"
        );
        // amendment 2, decision 1: a removed copy shows its source slate's options, as shown,
        // minus the right tool, and one extra in its place
        let by_uid: HashMap<&str, &Record5> = all.iter().map(|r| (r.r.uid.as_str(), r)).collect();
        let mut copies = 0;
        for r in all.iter().filter(|r| r.r.kind == "removed") {
            let from = r.from.as_deref().unwrap();
            let src = by_uid[format!("{}/{from}", r.r.request).as_str()];
            let g = src.r.target.iter().position(|&x| x == 1.0).unwrap();
            if src.r.tools.len() != r.r.tools.len() {
                // no distractor left (a small pool): the source slate without its gold, as v4
                let mut want = src.r.tools.clone();
                want.remove(g);
                assert_eq!(r.r.tools, want, "{}", r.r.uid);
                continue;
            }
            for i in (0..r.r.tools.len()).filter(|&i| i != g) {
                assert_eq!(r.r.tools[i], src.r.tools[i]);
                assert_eq!(
                    sky_infer::route::plain_option(&r.r.opts[i]),
                    sky_infer::route::plain_option(&src.r.opts[i]),
                    "{}",
                    r.r.uid
                );
                assert_eq!(r.roles[i], src.roles[i]);
            }
            assert_eq!(r.roles[g], DRAWN);
            assert!(r.cell_level >= src.cell_level);
            copies += 1;
        }
        assert!(copies > 100, "{copies}");
        // amendment 2, decision 3: without a tool gold, the hard slate holds the request's own
        // tools first, as v2's did
        let mut nocall = 0;
        for p in plans.iter().filter(|p| p.gold.is_none()) {
            let p2 = v2_plans.iter().find(|x| x.id == p.id).unwrap();
            let own: HashSet<&str> = by_req[p.id.as_str()]
                .own
                .iter()
                .map(|&t| l.tools[t].id.as_str())
                .collect();
            for (t, d) in p.hard.tools.iter().zip(&p.hard.drawn) {
                assert_eq!(!d, own.contains(t.as_str()), "{}", p.id);
            }
            for t in p2.hard.iter().filter(|t| own.contains(t.as_str())) {
                assert!(p.hard.tools.contains(t), "{}", p.id);
            }
            nocall += 1;
        }
        assert!(nocall > 10);
        // amendment 2, decision 2: C6 judges the cells of at least 200 drawn slates
        for c in srep["C6 cells"].as_array().unwrap() {
            let drawn = c["counts"]["drawn slates"].as_u64().unwrap();
            assert_eq!(c["judged"].as_bool().unwrap(), drawn >= 200, "{c}");
        }
        // the request's own tools of v2's natural slates are kept, at v2's sizes
        let ix: HashMap<&str, &Plan> = v2_plans.iter().map(|p| (p.id.as_str(), p)).collect();
        for p in &plans {
            let p2 = ix[p.id.as_str()];
            let own: HashSet<&str> = by_req[p.id.as_str()]
                .own
                .iter()
                .map(|&t| l.tools[t].id.as_str())
                .collect();
            for t in p2.natural.iter().filter(|t| own.contains(t.as_str())) {
                assert!(p.natural.tools.contains(t), "{}", p.id);
            }
            assert_eq!(p.natural.tools.len(), p2.natural.len());
            assert_eq!(p.hard.tools.len(), p2.hard.len());
        }
        // the removed slates are drawn around the absent gold's cell: D5's extra matches the
        // gold's key (the natural slate's text) at the copy's level
        let mut extras = 0;
        for p in &plans {
            let (Some(g), Some(rm)) = (&p.gold, &p.natural_removed) else {
                continue;
            };
            let gp = p.natural.tools.iter().position(|t| t == g).unwrap();
            if rm.level >= surface::L6 || rm.tools.len() != p.natural.tools.len() {
                continue;
            }
            let shown = |id: &str| {
                let t = by_id[id];
                let d = match rw.text.get(id) {
                    Some(d) if v4::rewrite_draw(7, &p.id, "natural", id) => d.clone(),
                    _ => t.desc.clone(),
                };
                cuts.key(&t.name, &d)
            };
            assert!(
                surface::matches(shown(g), shown(&rm.tools[gp]), rm.level),
                "{}",
                p.id
            );
            assert!(rm.drawn[gp] && rm.level >= p.natural.level);
            extras += 1;
        }
        assert!(extras > 100);
        // the counts: the dropped requests and the replaced distractors
        let cnt = srep["counts"].as_object().unwrap();
        assert!(cnt
            .keys()
            .any(|k| k.ends_with("dropped, the teacher says the gold cannot")));
        assert!(cnt.keys().any(|k| k.ends_with("distractors replaced")));
        assert!(
            srep["teacher (4.2, 9)"]["requests dropped"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert!(srep["C6 cells"].as_array().unwrap().len() >= 14);

        // C6 at plan time and after the build count the same slates when the teacher drops no
        // request and names no distractor
        let rep_c = v5_run(&v2, &v4d, &c);
        stage_items(&c);
        fn_answers(&c, true);
        let srep_c = v5_slates(&v2, &v4d, &c, false).unwrap();
        assert_eq!(rep_c["C6 cells"], srep_c["C6 cells"]);

        // v5_tools: v_tools' requests, 8 options, 20% removed, the sources' texts, the levels
        let vt: Vec<Record5> = read_lines(&a.join("val/v5_tools.jsonl")).unwrap();
        let v2t: Vec<RouteRecord> = read_lines(&v2.join("val/v_tools.jsonl")).unwrap();
        assert_eq!(
            vt.iter().map(|r| &r.r.request).collect::<Vec<_>>(),
            v2t.iter().map(|r| &r.request).collect::<Vec<_>>()
        );
        assert!(vt.iter().all(|r| r.r.opts.len() == slates::TEST_TOOLS + 1));
        let rem = vt.iter().filter(|r| r.r.kind == "removed").count();
        assert_eq!(rem, (vt.len() as f64 * 0.2).round() as usize);
        for r in &vt {
            for (o, id) in r.r.opts.iter().zip(&r.r.tools) {
                assert_eq!(o, &by_id[id.as_str()].option());
            }
        }
        // the probe copies and the model's copies
        let rwf: Vec<Record5> = read_lines(&a.join("val/v5_tools_rw.jsonl")).unwrap();
        assert!(!rwf.is_empty() && rwf.iter().all(|r| r.r.follow.is_some()));
        let mc: Vec<Value> = read_lines(&a.join("val/v5_tools_masked.model.jsonl")).unwrap();
        assert!(mc
            .iter()
            .all(|r| r["opts"][0].as_str().unwrap().starts_with("score ")
                && r["cell_level"].is_u64()));
        assert!(a.join("val/v_tools.model.jsonl").exists());
        assert!(a.join("val/r_tools_renamed.model.jsonl").exists());
        // the training pools of N2
        let pj = super::super::read_json(&a.join(POOLS5)).unwrap();
        assert!(pj[TOOLS].as_array().unwrap().len() > 100);
        assert_eq!(pj[INTENTS].as_array().unwrap().len(), 6);
        // never into v2's or v4's folder; a request embedding file of another plan is refused
        assert!(plan(&opts(&v2, &v4d, &v2)).is_err());
        assert!(plan(&opts(&v2, &v4d, &v4d)).is_err());
        write_json(
            &a.join(REQ_META),
            &json!({"source_sha256": "0", "ids_sha256": "0"}),
        )
        .unwrap();
        assert!(plan(&opts(&v2, &v4d, &a)).is_err());
        std::fs::remove_dir_all(root).ok();
    }
}
