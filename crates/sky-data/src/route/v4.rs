//! The data of s6-route v4, the router retrained without its shortcut
//! (`docs/s6-route4-prereg.md`, section 4). v2's build folder is read, never written: its stage
//! files, embeddings, clusters, splits, caps, plans of the hard slates and teacher answers are
//! kept (4.1); everything v4 writes goes to its own build folder.
//!
//! 1. [`plan`] (`sky-data route v4-plan --v2 <v2 build> --out <v4 build>`): v2's split recomputed
//!    and checked against v2's validation slates; for every training and validation request the
//!    natural slate drawn again by D7 on a stream of its own, v2's hard slate with the
//!    false-negative answers applied by their ids, the distractor that takes the gold's place in
//!    each removed copy (D5), the renamed slate by the test-slate rule (D3) and its removed twin;
//!    the validation slates `v4_tools` and `v4_tools_masked` (seed 1237, the requests of
//!    `v_tools`); the items of the one teacher job (`rewrite-description`, every tool of the
//!    training and validation pools and of the intents that has no rewrite yet, 4.2), in the
//!    format `scripts/route-teacher.sh` reads (`BUILD=<v4 build>`).
//! 2. The teacher (the lead runs it): `BUILD=<v4 build> scripts/route-teacher.sh
//!    rewrite-description loop`.
//! 3. [`slates`] (`sky-data route v4-slates --v2 <v2 build> --out <v4 build>`): the records of
//!    every kind with D1 (each tool option's description the rewrite with probability 0.5, by
//!    `route/rewrite/<request>/<kind>/<tool>`, never reading the gold), *none* on 20% of the
//!    renamed slates and of the other slates of each group and split, rendered by D4 (each tool
//!    option `score <s> rank <r>; <name>: <description>`, the state the request alone); the
//!    probe copies `v4_tools_rw` and `v4_tools_masked_rw` (4.5) and the model's copies of the
//!    validation files.
//!
//! The fill of D7 draws first among the pool's tools of the right tool's own source (the
//! request's source: for a request without a tool gold, the source of its own tools), then from
//! the whole pool; the hard slates keep v2's nearest tools. The remedy of 4.4 (`--remedy
//! surface-cell`, only after a data check has failed on a surface rule) puts the right tool's
//! surface cell first within its source. Every step prints counts only.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sky_infer::route::{options_with_scores, Render};

use super::build::{
    self, answers_path, fn_id, items_path, stage, Able, Assigned, Ctx, Description, Kept, Loaded,
    Plan, RewriteItem, RouteRecord, NONE_SHARE, REMOVED_SHARE, TEST_COSINE,
};
use super::slates::{self, Slate};
use super::{
    add, bump, cut_description, option_text, read_json, read_lines, write_json, Counts, Tool,
    INSTRUCTION, INTENTS, NONE_OPTION, WHEN2CALL,
};
use crate::record::write_jsonl;
use crate::rng::Rng;

/// v2's seed (its `BUILD.json`): v4 keeps v2's streams where 4.1 keeps them.
pub const V2_SEED: u64 = 1234;
/// The seed of the validation slates `v4_tools` (4.3).
pub const V4_VAL_SEED: u64 = 1237;
/// D1: the probability that a tool option's description is the rewrite.
pub const REWRITE_P: f64 = 0.5;
/// The name of the new validation slates (files `val/v4_tools*.jsonl`).
pub const V4_TOOLS: &str = "v4-tools";
/// The plan of v4 in its build folder.
pub const PLAN4: &str = "plan4.jsonl";

// ------------------------------------------------------------------------------- surface cells

/// The surface of a tool option that the request-blind rules read (study §7; the cell of 6.2,
/// item 5).
pub mod surface {
    /// The name's case style, as `scripts/s6-route4-blind.py` reads it: `spaced`, `snake` (a
    /// `_` or `-`), `camel`, `pascal` or `other`.
    pub fn name_style(name: &str) -> &'static str {
        if name.contains(' ') {
            return "spaced";
        }
        if name.contains('_') || name.contains('-') {
            return "snake";
        }
        let alnum = !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric());
        let first = name.chars().next();
        if alnum
            && first.is_some_and(|c| c.is_ascii_lowercase())
            && name.chars().any(|c| c.is_ascii_uppercase())
        {
            return "camel";
        }
        if alnum && first.is_some_and(|c| c.is_ascii_uppercase()) {
            // ([A-Z][a-z0-9]+){2,}: every capital followed by a lower-case letter or a digit
            let cs: Vec<char> = name.chars().collect();
            let caps = cs.iter().filter(|c| c.is_ascii_uppercase()).count();
            let followed = cs.iter().enumerate().all(|(i, c)| {
                !c.is_ascii_uppercase() || cs.get(i + 1).is_some_and(|n| !n.is_ascii_uppercase())
            });
            if caps >= 2 && followed {
                return "pascal";
            }
        }
        "other"
    }

    pub fn ends_with_period(desc: &str) -> bool {
        desc.ends_with('.')
    }

    pub fn words(desc: &str) -> usize {
        desc.split_whitespace().count()
    }

    /// The quintile (0 to 4) of a word count among `sorted` (the word counts of a catalogue,
    /// sorted): five times the share of the catalogue strictly below it, rounded down.
    pub fn quintile(sorted: &[usize], w: usize) -> usize {
        if sorted.is_empty() {
            return 0;
        }
        let below = sorted.partition_point(|&x| x < w);
        (5 * below / sorted.len()).min(4)
    }

    /// The cell of a tool: (name style, final period, length quintile).
    pub type Cell = (&'static str, bool, usize);

    pub fn cell(name: &str, desc: &str, sorted: &[usize]) -> Cell {
        (
            name_style(name),
            ends_with_period(desc),
            quintile(sorted, words(desc)),
        )
    }
}

// ---------------------------------------------------------------------------------- the plan

/// One training or validation request of v4 and its slates before the teacher (tool ids).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Plan4 {
    pub id: String,
    pub group: String,
    pub source: String,
    pub split: String,
    pub lang: String,
    pub user: String,
    pub kind: String,
    pub gold: Option<String>,
    /// D7: the natural slate drawn again (v2's size, the request's own tools, the fill of D7).
    pub natural: Vec<String>,
    /// D5: the distractor that takes the gold's place in the natural slate's removed copy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub natural_extra: Option<String>,
    /// v2's hard slate with the false-negative answers applied.
    pub hard: Vec<String>,
    /// D5: the distractor that takes the gold's place in the hard slate's removed copy (the next
    /// tool of v2's reserve, else the fill of D7).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hard_extra: Option<String>,
    /// D3: the renamed slate by the test-slate rule (7 tools, the gold among them).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub renamed: Vec<String>,
    /// The same rule's removed twin (7 tools, without the gold).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub renamed_removed: Vec<String>,
}

pub struct PlanOptions {
    /// v2's build folder (read only).
    pub v2: PathBuf,
    /// v4's build folder.
    pub out: PathBuf,
    /// v2's seed.
    pub seed: u64,
    /// The remedy of 4.4: the right tool's surface cell first within its source.
    pub remedy: bool,
}

fn read_or_empty<T: serde::de::DeserializeOwned>(p: &Path) -> Result<Vec<T>> {
    if p.exists() {
        read_lines(p)
    } else {
        Ok(Vec::new())
    }
}

fn refused_ids(answers: &Path) -> Result<HashSet<String>> {
    Ok(
        read_or_empty::<crate::distill::Refused>(&crate::distill::refused_path(answers))?
            .into_iter()
            .map(|r| r.id)
            .collect(),
    )
}

/// The pools of random tools, as v2's split step draws them.
struct Pools {
    train: Vec<usize>,
    val: Vec<usize>,
    w2c_val: Vec<usize>,
    intents: BTreeMap<String, Vec<usize>>,
}

impl Pools {
    fn new(a: &Assigned, kept: &[Kept]) -> Self {
        let (train, val) = (a.pool("train"), a.pool("val"));
        let mut w2c_val: Vec<usize> = train.iter().chain(val.iter()).copied().collect();
        w2c_val.sort_unstable();
        let mut intents: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for k in kept.iter().filter(|k| k.r.group == INTENTS) {
            intents
                .entry(k.r.source.clone())
                .or_default()
                .extend(k.gold);
        }
        for v in intents.values_mut() {
            v.sort_unstable();
            v.dedup();
        }
        Pools {
            train,
            val,
            w2c_val,
            intents,
        }
    }

    /// The pool of a request and its name.
    fn of(&self, k: &Kept) -> (String, &[usize]) {
        if k.r.group == INTENTS {
            return (
                format!("intents/{}", k.r.source),
                &self.intents[&k.r.source],
            );
        }
        match (k.r.group.as_str(), k.split.as_str()) {
            (WHEN2CALL, "val") => ("w2c-val/val".into(), &self.w2c_val),
            (_, "train") => ("tools/train".into(), &self.train),
            _ => ("tools/val".into(), &self.val),
        }
    }
}

/// The random fill's layers (D7, and the remedy's surface cells), cached per pool and source.
struct Layers<'a> {
    l: &'a Loaded,
    /// The sources in whose staged requests each tool appears.
    src: HashMap<String, HashSet<usize>>,
    by: HashMap<(String, String), Vec<usize>>,
    remedy: bool,
    /// The word counts of every tool the slates can show, sorted (the quintiles of the remedy).
    sorted: Vec<usize>,
}

impl<'a> Layers<'a> {
    fn new(l: &'a Loaded, pools: &Pools, remedy: bool) -> Self {
        let mut src: HashMap<String, HashSet<usize>> = HashMap::new();
        for r in &l.requests {
            let e = src.entry(r.source.clone()).or_default();
            for id in r.tools.iter().chain(r.gold.iter()).chain(r.exclude.iter()) {
                e.insert(l.index[id]);
            }
        }
        let mut all: BTreeSet<usize> = pools.train.iter().chain(&pools.val).copied().collect();
        all.extend(pools.intents.values().flatten().copied());
        let mut sorted: Vec<usize> = all
            .iter()
            .map(|&t| surface::words(&l.tools[t].desc))
            .collect();
        sorted.sort_unstable();
        Layers {
            l,
            src,
            by: HashMap::new(),
            remedy,
            sorted,
        }
    }

    fn cell(&self, t: usize) -> surface::Cell {
        let tool = &self.l.tools[t];
        surface::cell(&tool.name, &tool.desc, &self.sorted)
    }

    /// The pool's tools of `source`, in pool order.
    fn source(&mut self, pool_name: &str, pool: &[usize], source: &str) -> Vec<usize> {
        let key = (pool_name.to_string(), source.to_string());
        if let Some(v) = self.by.get(&key) {
            return v.clone();
        }
        let empty = HashSet::new();
        let s = self.src.get(source).unwrap_or(&empty);
        let v: Vec<usize> = pool.iter().copied().filter(|t| s.contains(t)).collect();
        self.by.insert(key, v.clone());
        v
    }

    /// The layers of a fill for a request of `source` whose right tool is `gold`: its source,
    /// then the pool (D7); with the remedy, the right tool's surface cell within its source first,
    /// then the nearest quintiles of its style and end, then its style.
    fn of(
        &mut self,
        pool_name: &str,
        pool: &[usize],
        source: &str,
        gold: Option<usize>,
    ) -> Vec<Vec<usize>> {
        let src = self.source(pool_name, pool, source);
        let mut out = Vec::new();
        if let (true, Some(g)) = (self.remedy, gold) {
            let (style, end, q) = self.cell(g);
            let cells: Vec<surface::Cell> = src.iter().map(|&t| self.cell(t)).collect();
            for d in 0..5usize {
                let v: Vec<usize> = src
                    .iter()
                    .zip(&cells)
                    .filter(|(_, c)| c.0 == style && c.1 == end && c.2.abs_diff(q) == d)
                    .map(|(&t, _)| t)
                    .collect();
                out.push(v);
            }
            out.push(
                src.iter()
                    .zip(&cells)
                    .filter(|(_, c)| c.0 == style)
                    .map(|(&t, _)| t)
                    .collect(),
            );
        }
        out.push(src);
        out.push(pool.to_vec());
        out
    }
}

fn refs(v: &[Vec<usize>]) -> Vec<&[usize]> {
    v.iter().map(Vec::as_slice).collect()
}

/// The slate `tools` with its gold replaced in its place by `extra` (D5); without an extra the
/// gold is dropped.
fn removed_copy(tools: &[usize], gold: Option<usize>, extra: Option<usize>) -> Vec<usize> {
    let mut t = tools.to_vec();
    if let Some(gp) = gold.and_then(|g| t.iter().position(|&x| x == g)) {
        match extra {
            Some(e) => t[gp] = e,
            None => {
                t.remove(gp);
            }
        }
    }
    t
}

/// The test distractor filter of v2 4.4 (not an excluded tool, not the gold's normalised name,
/// MiniLM cosine to the gold below 0.8).
fn test_ok<'a>(cx: &'a Ctx<'a>, k: &'a Kept) -> impl Fn(usize) -> bool + 'a {
    let l = cx.l;
    move |t: usize| {
        let Some(g) = k.gold else { return true };
        !k.exclude.contains(&t)
            && l.norm[t] != l.norm[g]
            && cx.cos(&l.minilm, t, g).is_none_or(|c| c < TEST_COSINE)
    }
}

/// v2's build, loaded and its split recomputed, checked against what v2 wrote.
struct V2 {
    l: Loaded,
    a: Assigned,
    kept: Vec<Kept>,
}

fn load_v2(v2: &Path, seed: u64) -> Result<V2> {
    let l = build::load(v2)?;
    let m = read_json(&v2.join("BUILD.json")).ok();
    if let Some(s) = m.as_ref().and_then(|m| m["seed"].as_u64()) {
        if s != seed {
            bail!("v2's BUILD.json says seed {s}, not {seed}");
        }
    }
    let a = build::assign(&l)?;
    let kept = a.kept();
    // the guard: the recomputed split gives v2's validation slates byte for byte (v2's own file,
    // or the hash BUILD.json recorded)
    let cx = Ctx { l: &l, seed };
    let val_all = build::tool_reqs(&kept, "val");
    let files = build::tool_slates(&cx, "v-tools", "val", &val_all, &a.pool("val"));
    let h = super::sha256_hex(&build::records_bytes(&files["v-tools"])?);
    let path = v2.join("val/v_tools.jsonl");
    let want = match m
        .as_ref()
        .and_then(|m| m["outputs"]["val/v_tools.jsonl"]["sha256"].as_str())
    {
        Some(x) => x.to_string(),
        None => super::sha256_file(&path).with_context(|| format!("hash {}", path.display()))?,
    };
    if h != want {
        bail!(
            "the recomputed split does not give v2's val/v_tools.jsonl (sha256 {h}, v2 {want}): \
             v4 needs v2's split as it was built"
        );
    }
    Ok(V2 { l, a, kept })
}

/// Step 1 of v4: the plan, the validation slates and the rewrite job's items.
pub fn plan(o: &PlanOptions) -> Result<Value> {
    if o.out == o.v2 {
        bail!(
            "v4 writes to its own folder, never to v2's ({})",
            o.v2.display()
        );
    }
    let V2 { l, a, kept } = load_v2(&o.v2, o.seed)?;
    let cx = Ctx {
        l: &l,
        seed: o.seed,
    };
    let pools = Pools::new(&a, &kept);
    let mut layers = Layers::new(&l, &pools, o.remedy);
    let by_id: HashMap<&str, &Kept> = kept.iter().map(|k| (k.r.id.as_str(), k)).collect();
    let plans: Vec<Plan> = read_lines(&stage(&o.v2).join("plan.jsonl"))?;
    let able_path = answers_path(&o.v2, "false-negatives");
    let able: HashMap<String, Vec<usize>> = read_or_empty::<Able>(&able_path)?
        .into_iter()
        .map(|a| (a.id, a.able))
        .collect();
    let fn_refused = refused_ids(&able_path)?;
    let mut counts = Counts::new();
    let mut out = Vec::with_capacity(plans.len());
    let ix = |v: &[String]| v.iter().map(|id| l.index[id]).collect::<Vec<_>>();
    let ids = |v: &[usize]| v.iter().map(|&t| l.tools[t].id.clone()).collect::<Vec<_>>();
    for p in &plans {
        let Some(k) = by_id.get(p.id.as_str()).copied() else {
            bail!(
                "v2's plan holds {} that the recomputed split does not keep",
                p.id
            );
        };
        let gold = p.gold.as_ref().map(|g| l.index[g]);
        if k.split != p.split || k.gold != gold {
            bail!("{}: v2's plan and the recomputed split differ", p.id);
        }
        let (gs, sp) = (&p.group, &p.split);
        // v2's hard slate and the false-negative filter, by the answer's id (v2 4.5)
        let mut hard = Slate::new(ix(&p.hard), gold);
        let mut reserve = ix(&p.reserve).into_iter();
        let fid = fn_id(p, &l.tools, &l.index);
        if let Some(ans) = able.get(&fid) {
            if let Some(gp) = hard.gold {
                if !ans.contains(&(gp + 1)) {
                    bump(
                        &mut counts,
                        format!("{gs} {sp}: dropped, the teacher says the gold cannot"),
                    );
                    continue;
                }
            }
            for (i, t) in hard.tools.clone().iter().enumerate() {
                if Some(i) == hard.gold || !ans.contains(&(i + 1)) {
                    continue;
                }
                bump(&mut counts, format!("{gs} {sp}: distractors replaced"));
                match reserve.by_ref().find(|r| !hard.tools.contains(r) && r != t) {
                    Some(r) => hard.tools[i] = r,
                    None => bump(
                        &mut counts,
                        format!("{gs} {sp}: no tool left to replace a distractor"),
                    ),
                }
            }
        } else if fn_refused.contains(&fid) {
            bump(
                &mut counts,
                format!("{gs} {sp}: refused by moderation, hard slate unfiltered"),
            );
        } else {
            bail!("{}: no false-negative answer in v2's build ({fid})", p.id);
        }
        let (pool_name, pool) = pools.of(k);
        let same_cluster = |t: usize| match gold {
            Some(g) => t == g || (a.cluster_of[g].is_some() && a.cluster_of[t] == a.cluster_of[g]),
            None => false,
        };
        let ok = |t: usize| {
            !k.exclude.contains(&t)
                && !same_cluster(t)
                && (k.r.group == INTENTS || a.allowed(&k.r.group, &k.split, t))
        };
        let lay = layers.of(&pool_name, pool, &k.r.source, gold);
        let lay = refs(&lay);
        // D7: the natural slate drawn again, v2's size, on a stream of its own
        let mut rng = cx.rng(&format!("route4/natural/{}", p.id));
        let natural = slates::around_layers(gold, &k.own, p.natural.len(), &lay, &mut rng, &ok);
        if natural.len() != p.natural.len() {
            bump(
                &mut counts,
                format!("{gs} {sp}: natural slates of another size than v2's"),
            );
        }
        // D5: one more distractor of each slate's rule in the gold's place
        let extra = |tools: &[usize], rng: &mut Rng| -> Option<usize> {
            let g = gold?;
            let mut t: Vec<usize> = tools.iter().copied().filter(|&x| x != g).collect();
            let want = t.len() + 1;
            slates::fill_layers(&mut t, want, &lay, rng, &|x| x != g && ok(x));
            (t.len() == want).then(|| *t.last().unwrap())
        };
        let natural_extra = extra(&natural, &mut rng);
        let hard_extra = match gold {
            None => None,
            Some(_) => reserve
                .by_ref()
                .find(|r| !hard.tools.contains(r))
                .or_else(|| {
                    bump(
                        &mut counts,
                        format!("{gs} {sp}: hard extra from the fill (reserve used up)"),
                    );
                    extra(
                        &hard.tools,
                        &mut cx.rng(&format!("route4/hard-extra/{}", p.id)),
                    )
                }),
        };
        if gold.is_some() && natural_extra.is_none() {
            bump(
                &mut counts,
                format!("{gs} {sp}: no distractor left for a natural removed copy"),
            );
        }
        if gold.is_some() && hard_extra.is_none() {
            bump(
                &mut counts,
                format!("{gs} {sp}: no distractor left for a hard removed copy"),
            );
        }
        // D3: the renamed slate by the test-slate rule on the request's pool, and its removed twin
        let (renamed, renamed_removed) = match gold {
            None => (Vec::new(), Vec::new()),
            Some(_) => {
                let own: Vec<usize> = k
                    .own
                    .iter()
                    .copied()
                    .filter(|t| pool.binary_search(t).is_ok())
                    .collect();
                let tok = test_ok(&cx, k);
                let rng = cx.rng(&format!("route4/renamed-slate/{}", p.id));
                let present =
                    slates::test_slate_layers(gold, &own, &lay, false, &mut rng.clone(), &tok);
                let removed =
                    slates::test_slate_layers(gold, &own, &lay, true, &mut rng.clone(), &tok);
                if present.tools.len() != slates::TEST_TOOLS
                    || removed.tools.len() != slates::TEST_TOOLS
                {
                    bump(
                        &mut counts,
                        format!("{gs} {sp}: renamed slates of fewer than 7 tools"),
                    );
                }
                (ids(&present.tools), ids(&removed.tools))
            }
        };
        bump(&mut counts, format!("{gs} {sp}: requests"));
        out.push(Plan4 {
            id: p.id.clone(),
            group: p.group.clone(),
            source: p.source.clone(),
            split: p.split.clone(),
            lang: p.lang.clone(),
            user: p.user.clone(),
            kind: p.kind.clone(),
            gold: p.gold.clone(),
            natural: ids(&natural),
            natural_extra: natural_extra.map(|t| l.tools[t].id.clone()),
            hard: ids(&hard.tools),
            hard_extra: hard_extra.map(|t| l.tools[t].id.clone()),
            renamed,
            renamed_removed,
        });
    }
    std::fs::create_dir_all(stage(&o.out))?;
    write_jsonl(&stage(&o.out).join(PLAN4), &out)?;

    // ---- the validation slates of 4.3: v_tools' requests, v2's test rule with D7's fill,
    // 20% removed, the sources' descriptions, seed 1237
    let vcx = Ctx {
        l: &l,
        seed: V4_VAL_SEED,
    };
    let reqs = build::tool_reqs(&kept, "val");
    let v2_val: Vec<RouteRecord> = read_lines(&o.v2.join("val/v_tools.jsonl"))?;
    let want: Vec<&str> = v2_val.iter().map(|r| r.request.as_str()).collect();
    let got: Vec<&str> = reqs.iter().map(|k| k.r.id.as_str()).collect();
    if want != got {
        bail!("v4_tools would not hold the requests of v_tools, in order");
    }
    let val_pool = &pools.val;
    let removed = slates::removed_set(
        reqs.len(),
        REMOVED_SHARE,
        &mut vcx.rng(&format!("route/{V4_TOOLS}/removed")),
    );
    let (mut main, mut masked) = (Vec::new(), Vec::new());
    for (i, k) in reqs.iter().enumerate() {
        let own: Vec<usize> = k
            .own
            .iter()
            .copied()
            .filter(|t| val_pool.binary_search(t).is_ok())
            .collect();
        let lay = layers.of("tools/val", val_pool, &k.r.source, k.gold);
        let mut rng = vcx.rng(&format!("route/{V4_TOOLS}/slate/{}", k.r.id));
        let s = slates::test_slate_layers(
            k.gold,
            &own,
            &refs(&lay),
            removed.contains(&i),
            &mut rng,
            &test_ok(&vcx, k),
        );
        let kind = if removed.contains(&i) {
            "removed"
        } else {
            "test"
        };
        main.push(vcx.record(V4_TOOLS, "val", k.r.id.clone(), kind, k, &s));
        let mut rng = vcx.rng(&format!("route/{V4_TOOLS}/masked/{}", k.r.id));
        let m = slates::masked(&s, |t| l.tools[t].desc.clone(), &mut rng);
        masked.push(vcx.record(
            &format!("{V4_TOOLS}-masked"),
            "val",
            k.r.id.clone(),
            "masked",
            k,
            &m,
        ));
    }
    let val_dir = o.out.join("val");
    write_jsonl(&val_dir.join("v4_tools.jsonl"), &main)?;
    write_jsonl(&val_dir.join("v4_tools_masked.jsonl"), &masked)?;

    // ---- the one teacher job (4.2): every tool of the pools and of the intents without a
    // rewrite in v2's build
    let v2_rw = answers_path(&o.v2, "rewrite-description");
    let mut have: HashSet<String> = read_or_empty::<Description>(&v2_rw)?
        .into_iter()
        .map(|d| d.id)
        .collect();
    have.extend(refused_ids(&v2_rw)?);
    let mut want: BTreeMap<usize, &str> = BTreeMap::new();
    for &t in &pools.train {
        want.insert(t, "train");
    }
    for &t in &pools.val {
        want.entry(t).or_insert("val");
    }
    let train_intents: HashSet<usize> = kept
        .iter()
        .filter(|k| k.r.group == INTENTS && k.split == "train")
        .filter_map(|k| k.gold)
        .collect();
    for &t in pools.intents.values().flatten() {
        want.entry(t).or_insert(if train_intents.contains(&t) {
            "train"
        } else {
            "val"
        });
    }
    let mut items: Vec<RewriteItem> = want
        .iter()
        .filter(|(t, _)| !have.contains(&l.tools[**t].id))
        .map(|(&t, split)| RewriteItem {
            id: l.tools[t].id.clone(),
            split: split.to_string(),
            name: l.tools[t].name.clone(),
            description: l.tools[t].desc.clone(),
        })
        .collect();
    items.sort_by(|a, b| a.id.cmp(&b.id));
    write_jsonl(&items_path(&o.out, "rewrite-description"), &items)?;
    let mut item_split = Counts::new();
    for it in &items {
        bump(&mut item_split, &it.split);
    }
    let report = json!({
        "about": "s6-route v4 (docs/s6-route4-prereg.md, section 4): the plan, counted before the teacher",
        "seed": o.seed,
        "v4_tools seed": V4_VAL_SEED,
        "remedy (surface cells first)": o.remedy,
        "plans": out.len(),
        "counts": counts,
        "v4_tools": {"slates": main.len(), "removed": removed.len()},
        "teacher items": {"rewrite-description": items.len(), "by split": item_split,
                          "tools of the pools and intents": want.len(),
                          "with a rewrite or refused in v2's build": want.len() - items.len()},
    });
    write_json(&o.out.join("counts/v4-plan.json"), &report)?;
    Ok(report)
}

// -------------------------------------------------------------------------------- the slates

pub struct SlatesOptions {
    pub v2: PathBuf,
    pub out: PathBuf,
    pub seed: u64,
    /// Without every rewrite: such a build is not trained on (smoke runs only).
    pub provisional: bool,
}

/// The rewrites of the build: v2's answers and v4's, cut by `cut_description`; and the tools
/// whose rewrite the moderation refused.
struct Rewrites {
    text: HashMap<String, String>,
    refused: HashSet<String>,
}

impl Rewrites {
    fn load(v2: &Path, out: &Path) -> Result<Self> {
        let mut text = HashMap::new();
        let mut refused = HashSet::new();
        for dir in [v2, out] {
            let p = answers_path(dir, "rewrite-description");
            for d in read_or_empty::<Description>(&p)? {
                text.insert(d.id, cut_description(&d.description));
            }
            refused.extend(refused_ids(&p)?);
        }
        Ok(Rewrites { text, refused })
    }
}

/// D1: whether the tool option `tool` of the `kind` slate of `request` shows the rewrite; the
/// draw reads the request, the kind and the tool, never the gold.
pub fn rewrite_draw(seed: u64, request: &str, kind: &str, tool: &str) -> bool {
    Rng::derived(seed, &format!("route/rewrite/{request}/{kind}/{tool}")).f64() < REWRITE_P
}

/// Step 3 of v4: the training and validation records (D1, D2, D4, D5), the probe copies and the
/// model's copies of the validation files.
pub fn slates(o: &SlatesOptions) -> Result<Value> {
    if o.out == o.v2 {
        bail!(
            "v4 writes to its own folder, never to v2's ({})",
            o.v2.display()
        );
    }
    let tools: Vec<Tool> = read_lines(&stage(&o.v2).join("tools.jsonl"))?;
    let index: HashMap<String, usize> = tools
        .iter()
        .enumerate()
        .map(|(i, t)| (t.id.clone(), i))
        .collect();
    let plans: Vec<Plan4> = read_lines(&stage(&o.out).join(PLAN4))?;
    let rw = Rewrites::load(&o.v2, &o.out)?;
    let items: Vec<RewriteItem> = read_or_empty(&items_path(&o.out, "rewrite-description"))?;
    let missing = items
        .iter()
        .filter(|i| !rw.text.contains_key(&i.id) && !rw.refused.contains(&i.id))
        .count();
    if missing > 0 && !o.provisional {
        bail!(
            "the teacher has not answered every item ({missing} rewrites missing): \
             BUILD={} scripts/route-teacher.sh rewrite-description loop, or --provisional (not trained on)",
            o.out.display()
        );
    }
    let mut counts = Counts::new();
    let mut shown: BTreeMap<String, Counts> = BTreeMap::new();
    // one record: names real or `tool_<k>` (a stream of the renamed kind), D1 on every tool
    // option, D4's rendering
    let mut rec = |p: &Plan4,
                   ts: &[String],
                   kind: &str,
                   uid: &str,
                   masked: Option<&mut Rng>|
     -> Result<RouteRecord> {
        let gold = p.gold.as_ref().and_then(|g| ts.iter().position(|t| t == g));
        let ks: Option<Vec<usize>> = masked.map(|rng| {
            let mut ks: Vec<usize> = (1..=99).collect();
            rng.shuffle(&mut ks);
            ks
        });
        let c = shown.entry(format!("{} {}", p.group, p.split)).or_default();
        let mut opts = Vec::with_capacity(ts.len() + 1);
        let mut flags = Vec::with_capacity(ts.len());
        for (i, id) in ts.iter().enumerate() {
            let t = &tools[*index
                .get(id)
                .with_context(|| format!("{}: unknown tool {id}", p.id))?];
            let mut desc = t.desc.clone();
            let mut flag = false;
            bump(c, "tool options");
            if rewrite_draw(o.seed, &p.id, kind, id) {
                bump(c, "drawn for the rewrite");
                if let Some(d) = rw.text.get(id) {
                    desc = d.clone();
                    flag = true;
                } else if rw.refused.contains(id) {
                    bump(c, "drawn, the rewrite refused: shown original");
                } else {
                    bump(c, "drawn, no rewrite (provisional): shown original");
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
        Ok(RouteRecord {
            src: format!("{}:{}", p.group, p.source),
            t: "choice".into(),
            ins: INSTRUCTION.into(),
            opts: model_opts,
            state: super::state_of(&p.user),
            target,
            split: p.split.clone(),
            uid: format!("{}/{uid}", p.id),
            kind: kind.into(),
            lang: p.lang.clone(),
            request: p.id.clone(),
            tools: ts.to_vec(),
            follow: None,
            rewritten: Some(flags),
        })
    };
    // D2: *none* on 20% of the renamed slates of each group and split (removed_set)
    let mut tool_gold: BTreeMap<(String, String), usize> = BTreeMap::new();
    for p in plans.iter().filter(|p| p.gold.is_some()) {
        *tool_gold
            .entry((p.group.clone(), p.split.clone()))
            .or_default() += 1;
    }
    let renamed_removed: BTreeMap<(String, String), HashSet<usize>> = tool_gold
        .iter()
        .map(|((g, s), &n)| {
            let mut rng = Rng::derived(o.seed, &format!("route4/renamed/removed/{g}/{s}"));
            (
                (g.clone(), s.clone()),
                slates::removed_set(n, NONE_SHARE, &mut rng),
            )
        })
        .collect();
    #[derive(Default)]
    struct Bucket {
        fixed: Vec<RouteRecord>,
        removed: Vec<RouteRecord>,
        spare: Vec<RouteRecord>,
        nocall: Vec<RouteRecord>,
        renamed: Vec<RouteRecord>,
    }
    let mut buckets: BTreeMap<(String, String), Bucket> = BTreeMap::new();
    let mut seen: BTreeMap<(String, String), usize> = BTreeMap::new();
    for p in &plans {
        let key = (p.group.clone(), p.split.clone());
        let b = buckets.entry(key.clone()).or_default();
        match &p.gold {
            None => {
                b.nocall
                    .push(rec(p, &p.natural, "natural", "natural", None)?);
                b.nocall.push(rec(p, &p.hard, "hard", "hard", None)?);
            }
            Some(g) => {
                b.fixed
                    .push(rec(p, &p.natural, "natural", "natural", None)?);
                b.fixed.push(rec(p, &p.hard, "hard", "hard", None)?);
                // D5: the removed copies keep their slate's option count
                let copy = |ts: &[String], extra: &Option<String>| -> Vec<String> {
                    let mut v = ts.to_vec();
                    if let Some(gp) = v.iter().position(|t| t == g) {
                        match extra {
                            Some(e) => v[gp] = e.clone(),
                            None => {
                                v.remove(gp);
                            }
                        }
                    }
                    v
                };
                let nat = copy(&p.natural, &p.natural_extra);
                let hard = copy(&p.hard, &p.hard_extra);
                let n = seen.entry(key.clone()).or_default();
                let order = if (*n).is_multiple_of(2) {
                    [(nat, false), (hard, true)]
                } else {
                    [(hard, false), (nat, true)]
                };
                let j = *n;
                *n += 1;
                for (ts, spare) in order {
                    if ts.is_empty() {
                        bump(
                            &mut counts,
                            format!("{} {}: no removed slate (a gold alone)", p.group, p.split),
                        );
                        continue;
                    }
                    if spare {
                        b.spare.push(rec(p, &ts, "removed", "removed-2", None)?);
                    } else {
                        b.removed.push(rec(p, &ts, "removed", "removed", None)?);
                    }
                }
                // D2, D3: renamed, by the test-slate rule; its removed twin on 20%
                let none = renamed_removed[&key].contains(&j);
                let ts = if none { &p.renamed_removed } else { &p.renamed };
                if ts.is_empty() {
                    bail!("{}: no renamed slate in the plan", p.id);
                }
                let mut rng = Rng::derived(o.seed, &format!("route4/renamed/{}", p.id));
                b.renamed
                    .push(rec(p, ts, "renamed", "renamed", Some(&mut rng))?);
            }
        }
    }
    // 4.4 of v2, on the slates that are not renamed; the renamed slates hold their own 20%
    let mut written = BTreeMap::new();
    let mut shares = BTreeMap::new();
    for ((group, split), mut b) in buckets {
        let t = b.fixed.len();
        let target = ((t as f64) * NONE_SHARE / (1.0 - NONE_SHARE)).round() as usize;
        let mut rng = Rng::derived(o.seed, &format!("route4/none-share/{group}/{split}"));
        let mut none: Vec<RouteRecord> = Vec::new();
        if b.nocall.len() >= target {
            rng.shuffle(&mut b.nocall);
            add(
                &mut counts,
                format!("{group} {split}: removed slates dropped"),
                b.removed.len(),
            );
            add(
                &mut counts,
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
                    &mut counts,
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
                    &mut counts,
                    format!("{group} {split}: removed slates added"),
                    more.min(b.spare.len()),
                );
                none.extend(b.spare.into_iter().take(more));
            }
        }
        let is_none = |r: &RouteRecord| r.target.last() == Some(&1.0);
        let renamed_none = b.renamed.iter().filter(|r| is_none(r)).count();
        let renamed_n = b.renamed.len();
        let mut all = b.fixed;
        all.extend(none);
        let other_n = all.len();
        let other_none = all.iter().filter(|r| is_none(r)).count();
        all.extend(b.renamed);
        all.sort_by(|a, b| a.uid.cmp(&b.uid));
        let n_none = all.iter().filter(|r| is_none(r)).count();
        let mut kinds: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for r in &all {
            let e = kinds.entry(r.kind.clone()).or_default();
            e.0 += 1;
            e.1 += is_none(r) as usize;
        }
        shares.insert(
            format!("{group} {split}"),
            json!({
                "slates": all.len(), "none gold": n_none, "share": n_none as f64 / all.len().max(1) as f64,
                "renamed": {"slates": renamed_n, "none gold": renamed_none, "share": renamed_none as f64 / renamed_n.max(1) as f64},
                "not renamed": {"slates": other_n, "none gold": other_none, "share": other_none as f64 / other_n.max(1) as f64},
                "by kind (slates, none gold)": kinds,
            }),
        );
        let path = o
            .out
            .join("records")
            .join(&group)
            .join(format!("{split}.jsonl"));
        written.insert(format!("{group}/{split}"), write_jsonl(&path, &all)?);
    }

    // ---- 4.5: the probe copies of the validation slates, and the model's copies (D4)
    let val = o.out.join("val");
    let main: Vec<RouteRecord> = read_lines(&val.join("v4_tools.jsonl"))?;
    let masked: Vec<RouteRecord> = read_lines(&val.join("v4_tools_masked.jsonl"))?;
    if main.len() != masked.len()
        || main
            .iter()
            .zip(&masked)
            .any(|(a, b)| a.request != b.request)
    {
        bail!("v4_tools and v4_tools_masked do not hold the same slates");
    }
    let (mut main_rw, mut masked_rw) = (Vec::new(), Vec::new());
    let mut probe = Counts::new();
    for (r, m) in main.iter().zip(&masked) {
        let g = r
            .target
            .iter()
            .position(|&x| x == 1.0)
            .unwrap_or(r.tools.len());
        let cand: Vec<usize> = (0..r.tools.len())
            .filter(|&i| i != g && rw.text.contains_key(&r.tools[i]))
            .collect();
        if cand.is_empty() {
            bump(
                &mut probe,
                "slates without a rewritten distractor (left out)",
            );
            continue;
        }
        let mut rng = Rng::derived(V4_VAL_SEED, &format!("route/{V4_TOOLS}-rw/{}", r.request));
        let d = cand[rng.below(cand.len())];
        let desc = &rw.text[&r.tools[d]];
        for (src, dst) in [(r, &mut main_rw), (m, &mut masked_rw)] {
            let mut x = src.clone();
            let name = sky_infer::route::option_name(&src.opts[d]).to_string();
            x.opts[d] = option_text(&name, desc);
            x.follow = Some(d);
            x.src = format!("{}-rw", src.src);
            dst.push(x);
        }
        bump(
            &mut probe,
            if g < r.tools.len() {
                "tool-gold slates"
            } else {
                "none-gold slates"
            },
        );
    }
    write_jsonl(&val.join("v4_tools_rw.jsonl"), &main_rw)?;
    write_jsonl(&val.join("v4_tools_masked_rw.jsonl"), &masked_rw)?;
    for f in [
        "v4_tools",
        "v4_tools_masked",
        "v4_tools_rw",
        "v4_tools_masked_rw",
    ] {
        build::render_with(
            &val.join(format!("{f}.jsonl")),
            &val.join(format!("{f}.model.jsonl")),
            Render::ScoresInOptions,
        )?;
    }
    // v2's v_tools and v_tools_masked, read by v4's pilot as reported readings (5): D4 copies
    for f in ["v_tools", "v_tools_masked"] {
        build::render_with(
            &o.v2.join("val").join(format!("{f}.jsonl")),
            &val.join(format!("{f}.model.jsonl")),
            Render::ScoresInOptions,
        )?;
    }
    // the renamed and natural validation records of the tools group (4.5)
    if let Ok(recs) = read_lines::<RouteRecord>(&o.out.join("records/tools/val.jsonl")) {
        for kind in ["renamed", "natural"] {
            let v: Vec<&RouteRecord> = recs.iter().filter(|r| r.kind == kind).collect();
            write_jsonl(&val.join(format!("r_tools_{kind}.model.jsonl")), &v)?;
        }
    }
    let report = json!({
        "about": "s6-route v4 (docs/s6-route4-prereg.md, section 4): the records, counted before any training",
        "seed": o.seed,
        "provisional": o.provisional,
        "rewrites missing": missing,
        "records": written,
        "none share": shares,
        "rewrites shown (D1), by group and split": shown,
        "probe copies (4.5)": probe,
        "counts": counts,
    });
    write_json(&o.out.join("counts/v4-slates.json"), &report)?;
    Ok(report)
}

/// The tools of `tools` (the slate's order) with the gold replaced in place by `extra`.
pub fn d5_copy(tools: &[usize], gold: Option<usize>, extra: Option<usize>) -> Vec<usize> {
    removed_copy(tools, gold, extra)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route::build::tests::{fixture, hashes, run};
    use crate::route::TOOLS;

    /// The real v2 build (`SKY_ROUTE_V2=<v2 build>`; read only, nothing written): its split
    /// recomputed gives its validation slates, and its stage files, plans and answers have the
    /// fields v4 reads. `cargo test -p sky-data --lib real_v2 -- --ignored`.
    #[test]
    #[ignore]
    fn real_v2_build_is_read() {
        let Ok(dir) = std::env::var("SKY_ROUTE_V2") else {
            return;
        };
        let dir = PathBuf::from(dir);
        let v = load_v2(&dir, V2_SEED).unwrap();
        let plans: Vec<Plan> = read_lines(&stage(&dir).join("plan.jsonl")).unwrap();
        let by: HashSet<&str> = v.kept.iter().map(|k| k.r.id.as_str()).collect();
        assert!(plans.iter().all(|p| by.contains(p.id.as_str())));
        let ans: Vec<Description> = read_lines(&answers_path(&dir, "rewrite-description")).unwrap();
        let able: Vec<Able> = read_lines(&answers_path(&dir, "false-negatives")).unwrap();
        let refused = refused_ids(&answers_path(&dir, "false-negatives")).unwrap();
        println!(
            "plans {} kept {} rewrites {} false-negative answers {} refused {}",
            plans.len(),
            v.kept.len(),
            ans.len(),
            able.len(),
            refused.len()
        );
        let index = &v.l.index;
        let ids: HashSet<String> = able.iter().map(|a| a.id.clone()).collect();
        let found = plans
            .iter()
            .filter(|p| {
                let id = fn_id(p, &v.l.tools, index);
                ids.contains(&id) || refused.contains(&id)
            })
            .count();
        assert_eq!(found, plans.len(), "false-negative answers found by id");
    }

    #[test]
    fn name_styles_and_quintiles() {
        use surface::*;
        for (n, s) in [
            ("Get Guardian Raids", "spaced"),
            ("get_weather", "snake"),
            ("madlibs-diceware", "snake"),
            ("getStockPrice", "camel"),
            ("GetStockPrice", "pascal"),
            ("GETData", "other"),
            ("Weather", "other"),
            ("weather", "other"),
            ("tool2", "other"),
        ] {
            assert_eq!(name_style(n), s, "{n}");
        }
        let sorted: Vec<usize> = (1..=10).collect();
        assert_eq!(quintile(&sorted, 1), 0);
        assert_eq!(quintile(&sorted, 3), 1);
        assert_eq!(quintile(&sorted, 10), 4);
        assert_eq!(quintile(&sorted, 99), 4);
        assert_eq!(quintile(&[], 3), 0);
        assert_eq!(cell("get_x", "Does a thing.", &sorted), ("snake", true, 1));
        assert_eq!(d5_copy(&[4, 2, 9], Some(2), Some(7)), vec![4, 7, 9]);
        assert_eq!(d5_copy(&[4, 2, 9], Some(2), None), vec![4, 9]);
    }

    #[test]
    fn rewrite_draws_are_seeded_and_near_half() {
        let n = 4000;
        let k = (0..n)
            .filter(|i| rewrite_draw(1234, &format!("glaive:{i}"), "natural", "t0"))
            .count();
        assert!((k as f64 / n as f64 - 0.5).abs() < 0.03, "{k}");
        assert_eq!(
            rewrite_draw(1234, "a", "hard", "t1"),
            rewrite_draw(1234, "a", "hard", "t1")
        );
    }

    /// The teacher's answers of v2's fixture build: four rewrite items of five answered;
    /// false-negative
    /// answers that name the gold able, except some that say it cannot and some that name a
    /// distractor too.
    fn v2_answers(dir: &Path) {
        let items: Vec<RewriteItem> = read_lines(&items_path(dir, "rewrite-description")).unwrap();
        // every fifth tool without a rewrite in v2's build: v4's job asks for it
        let ans: Vec<Value> = items
            .iter()
            .enumerate()
            .filter(|(j, _)| j % 5 != 0)
            .map(|(_, i)| json!({"id": i.id, "description": format!("Returns the result of {}. Extra.", i.name)}))
            .collect();
        write_jsonl(&answers_path(dir, "rewrite-description"), &ans).unwrap();
        let tools: Vec<Tool> = read_lines(&stage(dir).join("tools.jsonl")).unwrap();
        let index: HashMap<String, usize> = tools
            .iter()
            .enumerate()
            .map(|(i, t)| (t.id.clone(), i))
            .collect();
        let plans: Vec<Plan> = read_lines(&stage(dir).join("plan.jsonl")).unwrap();
        let mut able = Vec::new();
        for (j, p) in plans.iter().enumerate() {
            let id = fn_id(p, &tools, &index);
            let gp = p
                .gold
                .as_ref()
                .and_then(|g| p.hard.iter().position(|t| t == g));
            let mut a: Vec<usize> = gp.map(|g| g + 1).into_iter().collect();
            if j % 17 == 3 {
                a.clear(); // the gold cannot: the request is dropped
            } else if j % 13 == 5 {
                if let Some(i) = (0..p.hard.len()).find(|&i| Some(i) != gp) {
                    a.push(i + 1); // a distractor can: it is replaced
                }
            }
            a.sort_unstable();
            able.push(json!({"id": id, "able": a}));
        }
        write_jsonl(&answers_path(dir, "false-negatives"), &able).unwrap();
    }

    /// v4's answers: every item but the first refused.
    fn v4_answers(out: &Path) -> String {
        let items: Vec<RewriteItem> = read_lines(&items_path(out, "rewrite-description")).unwrap();
        let ans: Vec<Value> = items[1..]
            .iter()
            .map(|i| json!({"id": i.id, "description": format!("Returns what {} gives.", i.name)}))
            .collect();
        write_jsonl(&answers_path(out, "rewrite-description"), &ans).unwrap();
        let refused = vec![
            json!({"id": items[0].id, "src": "rewrite-description", "reason": "moderation (1301)"}),
        ];
        write_jsonl(
            &crate::distill::refused_path(&answers_path(out, "rewrite-description")),
            &refused,
        )
        .unwrap();
        items[0].id.clone()
    }

    fn v4(v2: &Path, out: &Path) {
        plan(&PlanOptions {
            v2: v2.into(),
            out: out.into(),
            seed: 7,
            remedy: false,
        })
        .unwrap();
    }

    fn v4_slates(v2: &Path, out: &Path, provisional: bool) -> Result<Value> {
        slates(&SlatesOptions {
            v2: v2.into(),
            out: out.into(),
            seed: 7,
            provisional,
        })
    }

    fn gold_of(r: &RouteRecord) -> usize {
        r.target.iter().position(|&x| x == 1.0).unwrap()
    }

    #[test]
    fn an_invented_v4_build() {
        let root = std::env::temp_dir().join(format!("sky-route-v4-{}", std::process::id()));
        let v2 = root.join("v2");
        fixture(&v2);
        run(&v2, 7);
        v2_answers(&v2);
        let before = hashes(&v2);
        let (a, b) = (root.join("a"), root.join("b"));
        v4(&v2, &a);
        // without v4's answers: only a provisional build
        assert!(v4_slates(&v2, &a, false).is_err());
        let refused = v4_answers(&a);
        v4_slates(&v2, &a, false).unwrap();
        v4(&v2, &b);
        v4_answers(&b);
        v4_slates(&v2, &b, false).unwrap();
        // v2's folder is never written
        assert_eq!(hashes(&v2), before);
        // determinism under the seed
        let hb = |d: &Path| {
            let mut h = hashes(d);
            h.retain(|k, _| !k.starts_with("teacher/"));
            h
        };
        assert_eq!(hb(&a), hb(&b));

        let tools: Vec<Tool> = read_lines(&stage(&v2).join("tools.jsonl")).unwrap();
        let by_id: HashMap<&str, &Tool> = tools.iter().map(|t| (t.id.as_str(), t)).collect();
        let plans2: Vec<Plan> = read_lines(&stage(&v2).join("plan.jsonl")).unwrap();
        let plans4: Vec<Plan4> = read_lines(&stage(&a).join(PLAN4)).unwrap();
        // the false-negative answers apply by their ids: the requests whose gold cannot are
        // dropped, the distractors the teacher names are replaced
        let kept4: HashSet<&str> = plans4.iter().map(|p| p.id.as_str()).collect();
        let mut dropped = 0;
        for (j, p) in plans2.iter().enumerate() {
            if j % 17 == 3 && p.gold.is_some() {
                assert!(!kept4.contains(p.id.as_str()), "{}", p.id);
                dropped += 1;
            }
        }
        assert!(dropped > 0);
        let p4: HashMap<&str, &Plan4> = plans4.iter().map(|p| (p.id.as_str(), p)).collect();
        let mut replaced = 0;
        for (j, p) in plans2.iter().enumerate() {
            if j % 13 == 5 && j % 17 != 3 && p.gold.is_some() && p.hard.len() > 1 {
                let q = p4[p.id.as_str()];
                assert_ne!(q.hard, p.hard, "{}", p.id);
                replaced += 1;
            } else if j % 17 != 3 && j % 13 != 5 {
                if let Some(q) = p4.get(p.id.as_str()) {
                    assert_eq!(q.hard, p.hard, "{}", p.id);
                }
            }
        }
        assert!(replaced > 0);

        let mut all: Vec<RouteRecord> = Vec::new();
        for g in [TOOLS, WHEN2CALL, INTENTS] {
            for s in ["train", "val"] {
                let recs: Vec<RouteRecord> =
                    read_lines(&a.join(format!("records/{g}/{s}.jsonl"))).unwrap();
                assert!(!recs.is_empty(), "{g} {s}");
                // D2: none on 20% of the renamed slates of each group and split
                let ren: Vec<&RouteRecord> = recs.iter().filter(|r| r.kind == "renamed").collect();
                let none = ren
                    .iter()
                    .filter(|r| gold_of(r) == r.opts.len() - 1)
                    .count();
                assert_eq!(none, (ren.len() as f64 * 0.2).round() as usize, "{g} {s}");
                // and 20% of the others, to the nearest slate
                let other: Vec<&RouteRecord> =
                    recs.iter().filter(|r| r.kind != "renamed").collect();
                let on = other
                    .iter()
                    .filter(|r| gold_of(r) == r.opts.len() - 1)
                    .count();
                assert!(
                    (on as f64 - 0.2 * other.len() as f64).abs() <= 1.0,
                    "{g} {s}: {on} of {}",
                    other.len()
                );
                all.extend(recs);
            }
        }
        let uids: HashSet<&str> = all.iter().map(|r| r.uid.as_str()).collect();
        assert_eq!(uids.len(), all.len());
        let (mut rg, mut ng, mut ro, mut no) = (0usize, 0usize, 0usize, 0usize);
        for r in &all {
            assert_eq!(r.opts.last().unwrap(), NONE_OPTION);
            assert_eq!(r.state, super::super::state_of(&r.state[6..]));
            assert!(!r.state.contains('\n'));
            let flags = r.rewritten.as_ref().unwrap();
            assert_eq!(flags.len(), r.tools.len());
            let g = gold_of(r);
            for (i, (o, id)) in r.opts.iter().zip(&r.tools).enumerate() {
                // D4: the score and rank first, then the option
                let plain = sky_infer::route::plain_option(o);
                assert_ne!(plain, o.as_str(), "{o}");
                let desc = plain.split_once(": ").unwrap().1;
                let t = by_id[id.as_str()];
                if flags[i] {
                    assert_ne!(desc, t.desc);
                } else {
                    assert_eq!(desc, t.desc);
                }
                if id == &refused {
                    assert!(!flags[i]);
                }
                if i == g {
                    (rg, ng) = (rg + flags[i] as usize, ng + 1);
                } else {
                    (ro, no) = (ro + flags[i] as usize, no + 1);
                }
            }
            // D3: 8 options in every renamed slate (the invented intents are 6: fewer there),
            // every name `tool_<k>`, distinct
            if r.kind == "renamed" && !r.src.starts_with("intents:") {
                assert_eq!(r.opts.len(), slates::TEST_TOOLS + 1, "{}", r.uid);
                let names: HashSet<&str> = r.opts[..r.opts.len() - 1]
                    .iter()
                    .map(|o| sky_infer::route::option_name(sky_infer::route::plain_option(o)))
                    .collect();
                assert_eq!(names.len(), slates::TEST_TOOLS);
                assert!(names.iter().all(|n| n.starts_with("tool_")));
            }
        }
        // D1: P(rewritten | right tool) and P(rewritten | other tool option), both near 0.5
        let (pr, po) = (rg as f64 / ng as f64, ro as f64 / no as f64);
        assert!((pr - po).abs() < 0.05, "{pr} {po}");
        assert!((po - 0.5).abs() < 0.05, "{po}");
        // D5: a removed slate has the option count of the slate it comes from
        let by_req: HashMap<(&str, &str), usize> = all
            .iter()
            .filter(|r| r.kind == "natural" || r.kind == "hard")
            .map(|r| ((r.request.as_str(), r.kind.as_str()), r.opts.len()))
            .collect();
        for r in all.iter().filter(|r| r.kind == "removed") {
            let n = by_req.get(&(r.request.as_str(), "natural")).copied();
            let h = by_req.get(&(r.request.as_str(), "hard")).copied();
            let p = p4[r.request.as_str()];
            let want: HashSet<usize> = [
                p.natural.len() + p.natural_extra.is_some() as usize,
                p.hard.len() + p.hard_extra.is_some() as usize,
            ]
            .into_iter()
            .collect();
            assert!(
                want.contains(&r.opts.len()),
                "{}: {} {n:?} {h:?}",
                r.uid,
                r.opts.len()
            );
        }
        // D7: When2Call's natural slates draw from When2Call's own tools first
        let reqs: Vec<super::super::Request> =
            read_lines(&stage(&v2).join("requests.jsonl")).unwrap();
        let w2c_tools: HashSet<&str> = reqs
            .iter()
            .filter(|r| r.source == "when2call")
            .flat_map(|r| r.tools.iter().chain(r.gold.iter()).map(String::as_str))
            .collect();
        let w2c_nat: Vec<&RouteRecord> = all
            .iter()
            .filter(|r| r.src == "when2call:when2call" && r.kind == "natural" && r.split == "train")
            .collect();
        assert!(!w2c_nat.is_empty());
        for r in &w2c_nat {
            assert!(
                r.tools.iter().all(|t| w2c_tools.contains(t.as_str())),
                "{}",
                r.uid
            );
        }

        // the validation slates: v_tools' requests, 8 options, 20% removed, the sources' texts
        let vt: Vec<RouteRecord> = read_lines(&a.join("val/v4_tools.jsonl")).unwrap();
        let v2t: Vec<RouteRecord> = read_lines(&v2.join("val/v_tools.jsonl")).unwrap();
        assert_eq!(
            vt.iter().map(|r| &r.request).collect::<Vec<_>>(),
            v2t.iter().map(|r| &r.request).collect::<Vec<_>>()
        );
        assert!(vt.iter().all(|r| r.opts.len() == slates::TEST_TOOLS + 1));
        let rem = vt.iter().filter(|r| r.kind == "removed").count();
        assert_eq!(rem, (vt.len() as f64 * 0.2).round() as usize);
        for r in &vt {
            for (o, id) in r.opts.iter().zip(&r.tools) {
                assert_eq!(o, &by_id[id.as_str()].option());
            }
        }
        // the probe copies: one distractor rewritten, `follow` on it, the same in both copies
        let rw: Vec<RouteRecord> = read_lines(&a.join("val/v4_tools_rw.jsonl")).unwrap();
        let mrw: Vec<RouteRecord> = read_lines(&a.join("val/v4_tools_masked_rw.jsonl")).unwrap();
        assert!(!rw.is_empty() && rw.len() == mrw.len());
        let vt_by: HashMap<&str, &RouteRecord> =
            vt.iter().map(|r| (r.request.as_str(), r)).collect();
        for (x, y) in rw.iter().zip(&mrw) {
            let d = x.follow.unwrap();
            assert_eq!(y.follow, Some(d));
            assert_ne!(d, gold_of(x));
            let orig = vt_by[x.request.as_str()];
            assert_ne!(x.opts[d], orig.opts[d]);
            assert_eq!(
                sky_infer::route::option_name(&x.opts[d]),
                sky_infer::route::option_name(&orig.opts[d])
            );
            for i in (0..x.opts.len()).filter(|&i| i != d) {
                assert_eq!(x.opts[i], orig.opts[i]);
            }
            assert!(y.opts[d].starts_with("tool_"));
        }
        // the model's copies: D4 on every file
        let mc: Vec<RouteRecord> = read_lines(&a.join("val/v4_tools_masked.model.jsonl")).unwrap();
        assert!(mc
            .iter()
            .all(|r| r.opts[0].starts_with("score ") && !r.state.contains('\n')));
        assert!(a.join("val/v_tools.model.jsonl").exists());
        assert!(a.join("val/r_tools_renamed.model.jsonl").exists());
        // the teacher job: every pool tool without a rewrite in v2's build, nothing else
        // (v2's answered rewrites)
        let v2_items: HashSet<String> =
            read_lines::<Description>(&answers_path(&v2, "rewrite-description"))
                .unwrap()
                .into_iter()
                .map(|i| i.id)
                .collect();
        let items: Vec<RewriteItem> = read_lines(&items_path(&a, "rewrite-description")).unwrap();
        assert!(!items.is_empty());
        assert!(items.iter().all(|i| !v2_items.contains(&i.id)));
        let shown: HashSet<&str> = all
            .iter()
            .chain(&vt)
            .flat_map(|r| r.tools.iter().map(String::as_str))
            .collect();
        for id in shown {
            assert!(
                v2_items.contains(id) || items.iter().any(|i| i.id == id),
                "{id} is shown and has no rewrite item"
            );
        }
        // never into v2's folder
        assert!(plan(&PlanOptions {
            v2: v2.clone(),
            out: v2.clone(),
            seed: 7,
            remedy: false
        })
        .is_err());
        // the remedy builds too
        let c = root.join("c");
        plan(&PlanOptions {
            v2: v2.clone(),
            out: c.clone(),
            seed: 7,
            remedy: true,
        })
        .unwrap();
        std::fs::remove_dir_all(root).ok();
    }
}
