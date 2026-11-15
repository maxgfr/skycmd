//! The steps of the build, in order (`scripts/build-route-data.sh` runs them):
//!
//! 1. [`intents()`]: the intent names the teacher describes (`describe-intent`, 4.5);
//! 2. [`requests()`]: every source read (`stage/tools.jsonl`, `stage/requests.jsonl`) and the
//!    texts the embedding script encodes (`stage/texts.jsonl`);
//! 3. `scripts/route-embed.py`: `emb/minilm.npy`, `emb/potion.npy`, `emb/meta.json`;
//! 4. [`split()`]: clusters, splits, caps, the test files (written once), the validation slates,
//!    the plan of the training slates, and the items of the two other teacher jobs;
//! 5. [`build_slates`]: the training and validation records with the teacher's answers;
//! 6. `BUILD.json`, by [`manifest()`].
//!
//! Nothing reads a test file after it is written: the steps that follow [`split()`] read the
//! stage files and the plan, which hold training and validation requests only.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::cluster::{self, Matrix};
use super::slates::{self, Slate};
use super::sources::{self, SourcePaths};
use super::{
    add, bump, norm_text, read_lines, sha256_file, sha256_hex, state_of, write_json, Counts,
    Request, Tool, INSTRUCTION, INTENTS, TEST, TOOLS, WHEN2CALL,
};
use crate::record::write_jsonl;
use crate::rng::Rng;

/// 4.2, item 2.
pub const CLUSTER_COSINE: f32 = 0.85;
/// 4.4: a test distractor's MiniLM cosine to the gold must stay below this.
pub const TEST_COSINE: f32 = 0.8;
/// 4.2, item 5.
pub const TRAIN_CAP: usize = 30;
pub const TEST_CAP: usize = 10;
/// 4.4 and 7.1.
pub const TEST_SLATES: usize = 2000;
pub const REMOVED_SHARE: f64 = 0.2;
pub const NONE_SHARE: f64 = 0.2;
/// 7.3: the gold-removed pairs and the swapped descriptions.
pub const PAIRS: usize = 500;
pub const SWAPS: usize = 500;
/// 7.1: the MASSIVE probe, per language.
pub const MASSIVE_PROBE: usize = 1000;
/// Nearest tools kept per gold for the hard slates (6 distractors at most, and the reserve the
/// false-negative filter replaces from).
pub const NEAREST: usize = 16;

/// A teacher item of `describe-intent`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct IntentItem {
    pub id: String,
    pub split: String,
    pub source: String,
    pub name: String,
}

/// A teacher item of `false-negatives`: the request and the tools of its hard slate, in order.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct FnItem {
    pub id: String,
    pub split: String,
    pub request: String,
    pub options: Vec<String>,
}

/// A teacher item of `rewrite-description`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct RewriteItem {
    pub id: String,
    pub split: String,
    pub name: String,
    pub description: String,
}

/// The teacher's answer to a `describe-intent` or `rewrite-description` item.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Description {
    pub id: String,
    pub description: String,
}

/// The teacher's answer to a `false-negatives` item: the options (1-based) it says can handle
/// the request.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Able {
    pub id: String,
    pub able: Vec<usize>,
}

/// One training or validation request and its slates before the teacher.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Plan {
    pub id: String,
    pub group: String,
    pub source: String,
    pub split: String,
    pub lang: String,
    pub user: String,
    pub kind: String,
    pub gold: Option<String>,
    pub natural: Vec<String>,
    pub hard: Vec<String>,
    pub reserve: Vec<String>,
}

/// The id of a request's false-negative item: the request's id and a hash of the request and of
/// its hard slate, so that an answer is never applied to another slate.
pub fn fn_id(p: &Plan, tools: &[Tool], index: &HashMap<String, usize>) -> String {
    let opts: Vec<String> = p.hard.iter().map(|id| tools[index[id]].option()).collect();
    let h = sha256_hex(format!("{}\n{}", p.user, opts.join("\n")).as_bytes());
    format!("{}#{}", p.id, &h[..12])
}

/// One record: the trainer's fields, then what the scorer and the baselines read.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct RouteRecord {
    pub src: String,
    pub t: String,
    pub ins: String,
    pub opts: Vec<String>,
    pub state: String,
    pub target: Vec<f32>,
    pub split: String,
    pub uid: String,
    pub kind: String,
    pub lang: String,
    pub request: String,
    /// The tool ids of the options before *none*.
    pub tools: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub follow: Option<usize>,
}

pub fn stage(out: &Path) -> PathBuf {
    out.join("stage")
}

fn teacher(out: &Path, job: &str) -> PathBuf {
    out.join("teacher").join(format!("{job}.jsonl"))
}

/// The items of a teacher job (`<out>/teacher/<job>.items.jsonl`).
pub fn items_path(out: &Path, job: &str) -> PathBuf {
    out.join("teacher").join(format!("{job}.items.jsonl"))
}

/// The answers of a teacher job (`<out>/teacher/<job>.jsonl`).
pub fn answers_path(out: &Path, job: &str) -> PathBuf {
    teacher(out, job)
}

fn read_or_empty<T: serde::de::DeserializeOwned>(p: &Path) -> Result<Vec<T>> {
    if p.exists() {
        read_lines(p)
    } else {
        Ok(Vec::new())
    }
}

// ----------------------------------------------------------------------------------- intents

/// Step 1: `<out>/teacher/describe-intent.items.jsonl`.
pub fn intents(p: &SourcePaths, out: &Path) -> Result<Value> {
    let items: Vec<IntentItem> = sources::intent_names(p)?
        .into_iter()
        .map(|(source, name)| IntentItem {
            id: format!("{source}:{name}"),
            // CLINC150 is tested only: its names (never a request) go to the teacher (4.5)
            split: if source == "clinc" { "names" } else { "train" }.into(),
            source,
            name,
        })
        .collect();
    let mut c = Counts::new();
    for i in &items {
        bump(&mut c, &i.source);
    }
    write_jsonl(&items_path(out, "describe-intent"), &items)?;
    Ok(json!({"describe-intent items": c}))
}

// ---------------------------------------------------------------------------------- requests

/// Step 2: every source read.
pub fn requests(p: &SourcePaths, out: &Path) -> Result<Value> {
    let mut p = p.clone();
    let answers: Vec<Description> = read_or_empty(&answers_path(out, "describe-intent"))?;
    p.descriptions = answers.into_iter().map(|d| (d.id, d.description)).collect();
    let (c, info) = sources::read_all(&p)?;
    let st = stage(out);
    std::fs::create_dir_all(&st)?;
    write_jsonl(&st.join("tools.jsonl"), c.tools.values())?;
    write_jsonl(&st.join("requests.jsonl"), &c.requests)?;
    write_json(&st.join("similar.json"), &json!(c.similar))?;
    write_jsonl(&st.join("metatool-tools.jsonl"), &c.metatool)?;
    // the texts to embed: the tools of the training groups and of the tool test
    let mut used = BTreeSet::new();
    for r in c.requests.iter().filter(|r| r.group != TEST) {
        used.extend(r.tools.iter().cloned());
        used.extend(r.gold.iter().cloned());
        used.extend(r.exclude.iter().cloned());
    }
    let texts: Vec<Value> = used
        .iter()
        .map(|id| json!({"id": id, "text": c.tools[id].option()}))
        .collect();
    write_jsonl(&st.join("texts.jsonl"), &texts)?;
    let mut by: BTreeMap<String, Counts> = BTreeMap::new();
    for r in &c.requests {
        bump(by.entry(r.group.clone()).or_default(), &r.source);
    }
    let report = json!({
        "read": c.counts,
        "requests by group and source": by,
        "tools": c.tools.len(),
        "texts to embed": texts.len(),
        "tools whose schema or description holds a key-like string": c.schema_keys.len(),
        "info": info,
    });
    write_json(&out.join("counts/requests.json"), &report)?;
    Ok(report)
}

// ------------------------------------------------------------------------------------- split

pub struct SplitOptions {
    pub out: PathBuf,
    pub seed: u64,
}

/// What the split step loads: the tool table, the requests, the embeddings by tool.
struct Loaded {
    tools: Vec<Tool>,
    index: HashMap<String, usize>,
    requests: Vec<Request>,
    minilm: Matrix,
    potion: Matrix,
    /// Row of each tool in the embedding files.
    row: Vec<Option<usize>>,
    norm: Vec<String>,
    stage: PathBuf,
}

fn load(out: &Path) -> Result<Loaded> {
    let st = stage(out);
    let tools: Vec<Tool> = read_lines(&st.join("tools.jsonl"))?;
    let index: HashMap<String, usize> = tools
        .iter()
        .enumerate()
        .map(|(i, t)| (t.id.clone(), i))
        .collect();
    let requests: Vec<Request> = read_lines(&st.join("requests.jsonl"))?;
    let texts: Vec<Value> = read_lines(&st.join("texts.jsonl"))?;
    let meta = super::read_json(&out.join("emb/meta.json"))?;
    let want = sha256_file(&st.join("texts.jsonl"))?;
    if meta["texts_sha256"].as_str() != Some(want.as_str()) {
        bail!("emb/meta.json was not written for this stage/texts.jsonl: run scripts/route-embed.py again");
    }
    let minilm = Matrix::read_npy(&out.join("emb/minilm.npy"))?;
    let potion = Matrix::read_npy(&out.join("emb/potion.npy"))?;
    if minilm.rows != texts.len() || potion.rows != texts.len() {
        bail!(
            "embeddings of {} and {} rows for {} texts",
            minilm.rows,
            potion.rows,
            texts.len()
        );
    }
    let mut row = vec![None; tools.len()];
    for (k, t) in texts.iter().enumerate() {
        let id = t["id"].as_str().unwrap_or("");
        let i = *index
            .get(id)
            .with_context(|| format!("texts.jsonl: unknown tool {id}"))?;
        row[i] = Some(k);
    }
    let norm = tools.iter().map(Tool::norm).collect();
    Ok(Loaded {
        tools,
        index,
        requests,
        minilm,
        potion,
        row,
        norm,
        stage: st,
    })
}

/// A request kept for a split, with its tools as indices.
#[derive(Clone, Debug)]
struct Kept {
    r: Request,
    split: String,
    gold: Option<usize>,
    own: Vec<usize>,
    exclude: HashSet<usize>,
}

/// Writes a test file once: a file that is there already must hold the same bytes (4.6, item 1;
/// section 12 forbids rebuilding one).
fn write_once(path: &Path, recs: &[RouteRecord]) -> Result<String> {
    let mut bytes = Vec::new();
    for r in recs {
        bytes.extend(serde_json::to_vec(r)?);
        bytes.push(b'\n');
    }
    let h = sha256_hex(&bytes);
    if path.exists() {
        if sha256_file(path)? != h {
            bail!(
                "{} exists and differs from what this build writes: a test file is built once \
                 (docs/s6-nano2-prereg.md 4.6, section 12); build into another folder to compare",
                path.display()
            );
        }
        return Ok(h);
    }
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    std::fs::write(path, bytes)?;
    Ok(h)
}

struct Ctx<'a> {
    l: &'a Loaded,
    seed: u64,
}

impl Ctx<'_> {
    fn text(&self, t: usize) -> String {
        self.l.tools[t].option()
    }

    fn rng(&self, label: &str) -> Rng {
        Rng::derived(self.seed, label)
    }

    fn cos(&self, m: &Matrix, a: usize, b: usize) -> Option<f32> {
        Some(cluster::dot(m.row(self.l.row[a]?), m.row(self.l.row[b]?)))
    }

    /// A record of a slate.
    fn record(
        &self,
        src: &str,
        split: &str,
        uid: String,
        kind: &str,
        k: &Kept,
        s: &Slate,
    ) -> RouteRecord {
        let (opts, target) = s.render(|t| self.text(t));
        RouteRecord {
            src: src.into(),
            t: "choice".into(),
            ins: INSTRUCTION.into(),
            opts,
            state: state_of(&k.r.user),
            target,
            split: split.into(),
            uid,
            kind: kind.into(),
            lang: k.r.lang.clone(),
            request: k.r.id.clone(),
            tools: s
                .tools
                .iter()
                .map(|&t| self.l.tools[t].id.clone())
                .collect(),
            follow: None,
        }
    }
}

/// The test-like slates of a set of `route/tools` requests (4.4): T-tools or the validation
/// slates; and the files of the reading checks (7.3).
fn tool_slates(
    cx: &Ctx,
    name: &str,
    split: &str,
    reqs: &[&Kept],
    pool: &[usize],
) -> BTreeMap<String, Vec<RouteRecord>> {
    let l = cx.l;
    let removed = slates::removed_set(
        reqs.len(),
        REMOVED_SHARE,
        &mut cx.rng(&format!("route/{name}/removed")),
    );
    let ok_for = |k: &Kept| {
        let gold = k.gold;
        let ex = k.exclude.clone();
        move |t: usize| {
            let Some(g) = gold else { return true };
            !ex.contains(&t)
                && l.norm[t] != l.norm[g]
                && cx.cos(&l.minilm, t, g).is_none_or(|c| c < TEST_COSINE)
        }
    };
    let mut main = Vec::new();
    let mut built: Vec<(usize, Slate)> = Vec::new();
    for (i, k) in reqs.iter().enumerate() {
        let ok = ok_for(k);
        let own: Vec<usize> = k
            .own
            .iter()
            .copied()
            .filter(|t| pool.binary_search(t).is_ok())
            .collect();
        let mut rng = cx.rng(&format!("route/{name}/slate/{}", k.r.id));
        let s = slates::test_slate(k.gold, &own, pool, removed.contains(&i), &mut rng, &ok);
        let kind = if removed.contains(&i) {
            "removed"
        } else {
            "test"
        };
        main.push(cx.record(name, split, k.r.id.clone(), kind, k, &s));
        built.push((i, s));
    }
    let present: Vec<usize> = built
        .iter()
        .filter(|(_, s)| s.gold.is_some())
        .map(|(i, _)| *i)
        .collect();
    // gold present and gold removed, the second built by the same rule
    let mut pairs = Vec::new();
    for j in slates::draw(
        present.len(),
        PAIRS,
        &mut cx.rng(&format!("route/{name}/pairs")),
    ) {
        let (i, s) = &built[present[j]];
        let k = reqs[*i];
        let ok = ok_for(k);
        let mut rng = cx.rng(&format!("route/{name}/pair/{}", k.r.id));
        // the gold replaced in its place by one more distractor of the same rule, the other
        // options where they were: an answer that keeps its position kept its answer (7.3)
        let mut tools = s.without_gold().tools;
        let want = tools.len() + 1;
        slates::fill(&mut tools, want, pool, &mut rng, &|t| {
            Some(t) != k.gold && ok(t)
        });
        let mut twin_tools = s.tools.clone();
        let gp = s.gold.unwrap();
        twin_tools[gp] = *tools.last().unwrap();
        if tools.len() < s.tools.len() {
            // no tool left in the pool: the gold's place stays empty
            twin_tools.remove(gp);
        }
        let twin = Slate::new(twin_tools, None);
        pairs.push(cx.record(
            &format!("{name}-pairs"),
            split,
            format!("{}/present", k.r.id),
            "test",
            k,
            s,
        ));
        pairs.push(cx.record(
            &format!("{name}-pairs"),
            split,
            format!("{}/removed", k.r.id),
            "removed",
            k,
            &twin,
        ));
    }
    let mut masked = Vec::new();
    for (i, s) in &built {
        let k = reqs[*i];
        let mut rng = cx.rng(&format!("route/{name}/masked/{}", k.r.id));
        let m = slates::masked(s, |t| l.tools[t].desc.clone(), &mut rng);
        masked.push(cx.record(
            &format!("{name}-masked"),
            split,
            k.r.id.clone(),
            "masked",
            k,
            &m,
        ));
    }
    let mut swapped = Vec::new();
    for j in slates::draw(
        present.len(),
        SWAPS,
        &mut cx.rng(&format!("route/{name}/swaps")),
    ) {
        let (i, s) = &built[present[j]];
        let k = reqs[*i];
        let mut rng = cx.rng(&format!("route/{name}/swap/{}", k.r.id));
        if let Some((w, d)) = slates::swapped(
            s,
            |t| l.tools[t].name.clone(),
            |t| l.tools[t].desc.clone(),
            &mut rng,
        ) {
            let mut r = cx.record(
                &format!("{name}-swapped"),
                split,
                k.r.id.clone(),
                "swapped",
                k,
                &w,
            );
            r.follow = Some(d);
            swapped.push(r);
        }
    }
    BTreeMap::from([
        (name.to_string(), main),
        (format!("{name}-pairs"), pairs),
        (format!("{name}-masked"), masked),
        (format!("{name}-swapped"), swapped),
    ])
}

/// Step 4: clusters, splits, caps, the test files, the validation slates, the plan of the
/// training slates and the items of the false-negative and rewrite jobs.
pub fn split(o: &SplitOptions) -> Result<Value> {
    let l = load(&o.out)?;
    let cx = Ctx {
        l: &l,
        seed: o.seed,
    };
    let n = l.tools.len();
    let mut counts = Counts::new();

    // 4.2: the tools of route/tools and of When2Call, clustered
    let mut in_u = vec![false; n];
    for r in l
        .requests
        .iter()
        .filter(|r| r.group == TOOLS || r.group == WHEN2CALL)
    {
        for id in r.tools.iter().chain(r.gold.iter()).chain(r.exclude.iter()) {
            in_u[l.index[id]] = true;
        }
    }
    let u: Vec<usize> = (0..n).filter(|&i| in_u[i]).collect();
    let rows: Vec<usize> = u
        .iter()
        .map(|&t| l.row[t].with_context(|| format!("tool {} was not embedded", l.tools[t].id)))
        .collect::<Result<_>>()?;
    let names: Vec<String> = u.iter().map(|&t| l.norm[t].clone()).collect();
    let rep = cluster::clusters(&l.minilm, &rows, &names, CLUSTER_COSINE);
    let mut cluster_of: Vec<Option<usize>> = vec![None; n];
    let mut members: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, &t) in u.iter().enumerate() {
        cluster_of[t] = Some(u[rep[i]]);
        members.entry(u[rep[i]]).or_default().push(t);
    }
    let mut csplit: BTreeMap<usize, &'static str> = BTreeMap::new();
    for (&c, ms) in &members {
        let key = ms.iter().map(|&t| l.norm[t].as_str()).min().unwrap_or("");
        csplit.insert(c, cluster::split_of_key(key));
    }
    let tsplit = |t: usize| cluster_of[t].map(|c| csplit[&c]);
    let allowed = |group: &str, split: &str, t: usize| -> bool {
        match (group, split) {
            (WHEN2CALL, "val") => matches!(tsplit(t), Some("train" | "val")),
            _ => {
                tsplit(t)
                    == Some(match split {
                        "train" => "train",
                        "val" => "val",
                        _ => "test",
                    })
            }
        }
    };

    // the requests and their splits
    let mut kept: Vec<Kept> = Vec::new();
    for r in &l.requests {
        let gold = r.gold.as_ref().map(|g| l.index[g]);
        let split = match (r.group.as_str(), &r.split) {
            (TOOLS, _) => tsplit(gold.unwrap()).unwrap().to_string(),
            (_, Some(s)) => s.clone(),
            _ => bail!("{}: no split", r.id),
        };
        if r.group == WHEN2CALL {
            if let Some(g) = gold {
                if !allowed(WHEN2CALL, &split, g) {
                    bump(
                        &mut counts,
                        format!(
                            "when2call {split}: gold in a {} cluster",
                            tsplit(g).unwrap_or("?")
                        ),
                    );
                    continue;
                }
            }
        }
        let own: Vec<usize> = r
            .tools
            .iter()
            .map(|id| l.index[id])
            .filter(|&t| r.group == TEST || r.group == INTENTS || allowed(&r.group, &split, t))
            .collect();
        kept.push(Kept {
            exclude: r.exclude.iter().map(|id| l.index[id]).collect(),
            r: r.clone(),
            split,
            gold,
            own,
        });
    }

    // 4.2 item 4, 4.3, 4.6 item 2: a text of a held-out side is dropped from the others
    // (test before validation before training)
    let side = |s: &str| match s {
        "test" => 2,
        "val" => 1,
        _ => 0,
    };
    let mut best: HashMap<String, u8> = HashMap::new();
    for k in &kept {
        let e = best.entry(norm_text(&k.r.user)).or_insert(0);
        *e = (*e).max(side(&k.split));
    }
    kept.retain(|k| {
        let keep = best[&norm_text(&k.r.user)] == side(&k.split);
        if !keep {
            bump(
                &mut counts,
                format!("{} {}: same text on a held-out side", k.r.group, k.split),
            );
        }
        keep
    });

    // 4.2 item 5: at most 30 training requests per cluster, 10 on the test (and validation) side
    let mut by_cluster: BTreeMap<(String, usize), Vec<usize>> = BTreeMap::new();
    for (i, k) in kept.iter().enumerate() {
        let Some(c) = k.gold.and_then(|g| cluster_of[g]) else {
            continue;
        };
        let capped = matches!(
            (k.r.group.as_str(), k.split.as_str()),
            (TOOLS | WHEN2CALL, "train") | (TOOLS, "val" | "test")
        );
        if capped {
            by_cluster.entry((k.split.clone(), c)).or_default().push(i);
        }
    }
    let mut drop = HashSet::new();
    let mut largest: BTreeMap<String, Vec<(usize, usize)>> = BTreeMap::new();
    for ((split, c), mut v) in by_cluster {
        let cap = if split == "train" {
            TRAIN_CAP
        } else {
            TEST_CAP
        };
        largest.entry(split.clone()).or_default().push((v.len(), c));
        v.sort_by_key(|&i| (sha256_hex(kept[i].r.user.as_bytes()), kept[i].r.id.clone()));
        for &i in &v[cap.min(v.len())..] {
            drop.insert(i);
            bump(
                &mut counts,
                format!("{} {split}: over the cluster cap", kept[i].r.group),
            );
        }
    }
    let kept: Vec<Kept> = kept
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !drop.contains(i))
        .map(|(_, k)| k)
        .collect();

    // the pools of random tools
    let pool_of = |split: &str| -> Vec<usize> {
        u.iter()
            .copied()
            .filter(|&t| tsplit(t) == Some(split))
            .collect()
    };
    let (train_pool, val_pool, test_pool) = (pool_of("train"), pool_of("val"), pool_of("test"));
    let w2c_val_pool: Vec<usize> = {
        let mut v: Vec<usize> = train_pool.iter().chain(val_pool.iter()).copied().collect();
        v.sort_unstable();
        v
    };

    // ---- the test files (written once) and the validation slates
    let tool_reqs = |split: &str| -> Vec<&Kept> {
        kept.iter()
            .filter(|k| k.r.group == TOOLS && k.split == split)
            .collect()
    };
    let test_all = tool_reqs("test");
    let drawn: Vec<&Kept> = slates::draw(test_all.len(), TEST_SLATES, &mut cx.rng("route/t-tools"))
        .into_iter()
        .map(|i| test_all[i])
        .collect();
    let mut test_files = tool_slates(&cx, "t-tools", "test", &drawn, &test_pool);
    let val_all = tool_reqs("val");
    let val_files = tool_slates(&cx, "v-tools", "val", &val_all, &val_pool);
    other_tests(&cx, &kept, &mut test_files)?;

    let test_dir = o.out.join("test");
    let mut test_hashes = BTreeMap::new();
    for (name, recs) in &test_files {
        let file = test_dir.join(format!("{}.jsonl", name.replace('-', "_")));
        test_hashes.insert(
            file.file_name().unwrap().to_string_lossy().to_string(),
            json!({"records": recs.len(), "sha256": write_once(&file, recs)?}),
        );
    }
    let mut val_counts = BTreeMap::new();
    for (name, recs) in &val_files {
        let file = o
            .out
            .join("val")
            .join(format!("{}.jsonl", name.replace('-', "_")));
        write_jsonl(&file, recs)?;
        val_counts.insert(name.clone(), recs.len());
    }

    // ---- the plan of the training and validation slates (4.4)
    let pool_for = |k: &Kept| -> &[usize] {
        match (k.r.group.as_str(), k.split.as_str()) {
            (WHEN2CALL, "val") => &w2c_val_pool,
            (_, "train") => &train_pool,
            _ => &val_pool,
        }
    };
    // intents: one pool per source, the intents of its training and validation requests
    let mut intent_pool: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for k in kept.iter().filter(|k| k.r.group == INTENTS) {
        intent_pool
            .entry(k.r.source.clone())
            .or_default()
            .extend(k.gold);
    }
    for v in intent_pool.values_mut() {
        v.sort_unstable();
        v.dedup();
    }
    let train_side: Vec<&Kept> = kept
        .iter()
        .filter(|k| k.r.group != TEST && k.split != "test")
        .collect();
    // the nearest tools of each gold (potion-base-2M cosine), among the other clusters of its pool
    let mut want: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
    let pool_name = |k: &Kept| -> String {
        if k.r.group == INTENTS {
            format!("intents/{}", k.r.source)
        } else {
            format!(
                "{}/{}",
                if k.r.group == WHEN2CALL && k.split == "val" {
                    "w2c-val"
                } else {
                    "tools"
                },
                k.split
            )
        }
    };
    for k in &train_side {
        let p = pool_name(k);
        let e = want.entry(p).or_default();
        match k.gold {
            Some(g) => {
                e.insert(g);
            }
            None => e.extend(k.own.iter().copied()),
        }
    }
    let mut near: BTreeMap<(String, usize), Vec<usize>> = BTreeMap::new();
    for (p, qs) in &want {
        let pool: Vec<usize> = if let Some(src) = p.strip_prefix("intents/") {
            intent_pool[src].clone()
        } else {
            match p.as_str() {
                "w2c-val/val" => w2c_val_pool.clone(),
                "tools/train" => train_pool.clone(),
                _ => val_pool.clone(),
            }
        };
        let rows_pool: Vec<usize> = pool.iter().filter_map(|&t| l.row[t]).collect();
        let tool_at: HashMap<usize, usize> = pool
            .iter()
            .filter_map(|&t| l.row[t].map(|r| (r, t)))
            .collect();
        let qrows: Vec<usize> = qs.iter().filter_map(|&t| l.row[t]).collect();
        let q_tool: HashMap<usize, usize> = qs
            .iter()
            .filter_map(|&t| l.row[t].map(|r| (r, t)))
            .collect();
        let same = |q: usize, p: usize| -> bool {
            let (a, b) = (q_tool[&q], tool_at[&p]);
            a == b || (cluster_of[a].is_some() && cluster_of[a] == cluster_of[b])
        };
        for (q, ns) in cluster::nearest_all(&l.potion, &qrows, &rows_pool, NEAREST, &same) {
            near.insert(
                (p.clone(), q_tool[&q]),
                ns.into_iter().map(|r| tool_at[&r]).collect(),
            );
        }
    }
    let mut plans = Vec::new();
    for k in &train_side {
        let p = pool_name(k);
        let pool: &[usize] = if k.r.group == INTENTS {
            &intent_pool[&k.r.source]
        } else {
            pool_for(k)
        };
        let gold = k.gold;
        let same_cluster = |t: usize| match gold {
            Some(g) => t == g || (cluster_of[g].is_some() && cluster_of[t] == cluster_of[g]),
            None => false,
        };
        let ok = |t: usize| {
            !k.exclude.contains(&t)
                && !same_cluster(t)
                && (k.r.group == INTENTS || allowed(&k.r.group, &k.split, t))
        };
        let mut rng = cx.rng(&format!("route/plan/{}", k.r.id));
        let (kn, kh) = (
            slates::draw_options(&mut rng),
            slates::draw_options(&mut rng),
        );
        let natural = slates::around(gold, &k.own, kn - 1, pool, &mut rng, &ok);
        let nearest: Vec<usize> = match gold {
            Some(g) => near.get(&(p.clone(), g)).cloned().unwrap_or_default(),
            None => {
                // no gold: the request's own tools, then their nearest tools in turn
                let lists: Vec<Vec<usize>> = k
                    .own
                    .iter()
                    .map(|t| near.get(&(p.clone(), *t)).cloned().unwrap_or_default())
                    .collect();
                let mut v: Vec<usize> = k.own.clone();
                for i in 0..NEAREST {
                    for li in &lists {
                        if let Some(&t) = li.get(i) {
                            if !v.contains(&t) {
                                v.push(t);
                            }
                        }
                    }
                }
                v
            }
        };
        let nearest: Vec<usize> = nearest.into_iter().filter(|&t| ok(t)).collect();
        let mut hard: Vec<usize> = gold.into_iter().collect();
        hard.extend(
            nearest
                .iter()
                .copied()
                .filter(|t| Some(*t) != gold)
                .take(kh - 1 - gold.is_some() as usize),
        );
        if hard.len() < kh - 1 {
            slates::fill(&mut hard, kh - 1, pool, &mut rng, &ok);
        }
        rng.shuffle(&mut hard);
        let reserve: Vec<usize> = nearest
            .iter()
            .copied()
            .filter(|t| !hard.contains(t))
            .collect();
        let ids = |v: &[usize]| v.iter().map(|&t| l.tools[t].id.clone()).collect::<Vec<_>>();
        plans.push(Plan {
            id: k.r.id.clone(),
            group: k.r.group.clone(),
            source: k.r.source.clone(),
            split: k.split.clone(),
            lang: k.r.lang.clone(),
            user: k.r.user.clone(),
            kind: k.r.kind.clone(),
            gold: gold.map(|g| l.tools[g].id.clone()),
            natural: ids(&natural),
            hard: ids(&hard),
            reserve: ids(&reserve),
        });
    }
    // the guard of 4.2 item 6: no tool of a test cluster in a training or validation slate,
    // and no training slate with a tool of a validation cluster
    for p in &plans {
        for id in p
            .natural
            .iter()
            .chain(&p.hard)
            .chain(&p.reserve)
            .chain(p.gold.iter())
        {
            let s = tsplit(l.index[id]);
            if s == Some("test") || (p.split == "train" && s == Some("val")) {
                bail!(
                    "{}: tool {id} of a {s:?} cluster in a {} slate",
                    p.id,
                    p.split
                );
            }
        }
    }
    for rec in &test_files["t-tools"] {
        for id in &rec.tools {
            if tsplit(l.index[id]) != Some("test") {
                bail!("t-tools: tool {id} is not of a test cluster");
            }
        }
    }
    write_jsonl(&stage(&o.out).join("plan.jsonl"), &plans)?;

    // the items of the two other teacher jobs (never a test request: 4.5)
    let fn_items: Vec<FnItem> = plans
        .iter()
        .map(|p| FnItem {
            id: fn_id(p, &l.tools, &l.index),
            split: p.split.clone(),
            request: p.user.clone(),
            options: p
                .hard
                .iter()
                .map(|id| l.tools[l.index[id]].option())
                .collect(),
        })
        .collect();
    write_jsonl(&items_path(&o.out, "false-negatives"), &fn_items)?;
    let mut rw: BTreeMap<String, String> = BTreeMap::new();
    for p in &plans {
        if let Some(g) = &p.gold {
            let e = rw.entry(g.clone()).or_insert_with(|| p.split.clone());
            if p.split == "train" {
                *e = "train".into();
            }
        }
    }
    let rw_items: Vec<RewriteItem> = rw
        .iter()
        .map(|(id, split)| {
            let t = &l.tools[l.index[id]];
            RewriteItem {
                id: id.clone(),
                split: split.clone(),
                name: t.name.clone(),
                description: t.desc.clone(),
            }
        })
        .collect();
    write_jsonl(&items_path(&o.out, "rewrite-description"), &rw_items)?;

    // ---- counts (test side: counts only)
    let mut requests_by: BTreeMap<String, Counts> = BTreeMap::new();
    for k in &kept {
        bump(
            requests_by
                .entry(format!("{} {}", k.r.group, k.split))
                .or_default(),
            &k.r.source,
        );
    }
    let mut clusters_by = Counts::new();
    let mut tools_by = Counts::new();
    for (&c, ms) in &members {
        bump(&mut clusters_by, csplit[&c]);
        add(&mut tools_by, csplit[&c], ms.len());
    }
    let mut big = BTreeMap::new();
    for (split, mut v) in largest {
        v.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let top: Vec<Value> = v
            .iter()
            .take(10)
            .map(|(nreq, c)| {
                let mut e = json!({"requests before the cap": nreq, "tools": members[c].len()});
                if split == "train" {
                    e["key"] = json!(members[c].iter().map(|&t| l.norm[t].clone()).min());
                }
                e
            })
            .collect();
        big.insert(split, top);
    }
    let intents_near_test = intent_pool
        .values()
        .flatten()
        .filter(|&&i| {
            test_pool.iter().any(|&t| {
                l.norm[t] == l.norm[i]
                    || cx.cos(&l.minilm, t, i).is_some_and(|c| c >= CLUSTER_COSINE)
            })
        })
        .count();
    let report = json!({
        "clusters by split": clusters_by,
        "tools by split (route/tools and When2Call)": tools_by,
        "largest clusters (training: with their key)": big,
        "requests by group, split and source": requests_by,
        "dropped": counts,
        "plans (training and validation requests)": plans.len(),
        "validation files": val_counts,
        "test files": test_hashes,
        "teacher items": {"false-negatives": fn_items.len(), "rewrite-description": rw_items.len()},
        "intents within the cluster rule of a test-cluster tool (reported, not dropped)": intents_near_test,
    });
    write_json(&o.out.join("counts/split.json"), &report)?;
    Ok(report)
}

/// The other test sources of 7.1, built by their rules; no compared system chooses a
/// distractor.
fn other_tests(
    cx: &Ctx,
    kept: &[Kept],
    files: &mut BTreeMap<String, Vec<RouteRecord>>,
) -> Result<()> {
    let l = cx.l;
    let of =
        |source: &str| -> Vec<&Kept> { kept.iter().filter(|k| k.r.source == source).collect() };
    let tool_set = |reqs: &[&Kept]| -> Vec<usize> {
        let mut v: Vec<usize> = reqs
            .iter()
            .flat_map(|k| k.gold.into_iter().chain(k.own.iter().copied()))
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    };

    // T-metatool: the query's tool, 3 of its similar-choices group where it has one, then random
    let similar: BTreeMap<String, Vec<String>> =
        serde_json::from_value(super::read_json(&stage_of(cx).join("similar.json"))?)?;
    let mt = of("metatool");
    // the tools of plugin_des.json, by name (MetaTool's names are its own)
    let mt_ids: BTreeSet<String> =
        super::read_lines::<Value>(&stage_of(cx).join("metatool-tools.jsonl"))?
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
    let by_name: HashMap<&str, usize> = mt_ids
        .iter()
        .map(|id| (l.tools[l.index[id]].name.as_str(), l.index[id]))
        .collect();
    let mut mt_pool: Vec<usize> = by_name.values().copied().collect();
    mt_pool.sort_unstable();
    let drawn = slates::draw(mt.len(), TEST_SLATES, &mut cx.rng("route/t-metatool"));
    let removed = slates::removed_set(
        drawn.len(),
        REMOVED_SHARE,
        &mut cx.rng("route/t-metatool/removed"),
    );
    let mut recs = Vec::new();
    for (j, &i) in drawn.iter().enumerate() {
        let k = mt[i];
        let g = k.gold.unwrap();
        let mut rng = cx.rng(&format!("route/t-metatool/slate/{}", k.r.id));
        let mut group: Vec<usize> = similar
            .get(&l.tools[g].name)
            .map(|v| {
                v.iter()
                    .filter_map(|n| by_name.get(n.as_str()).copied())
                    .filter(|&t| t != g)
                    .collect()
            })
            .unwrap_or_default();
        rng.shuffle(&mut group);
        group.truncate(3);
        let s = slates::test_slate(
            Some(g),
            &group,
            &mt_pool,
            removed.contains(&j),
            &mut rng,
            &|t| t != g,
        );
        let kind = if removed.contains(&j) {
            "removed"
        } else {
            "test"
        };
        recs.push(cx.record("t-metatool", "test", k.r.id.clone(), kind, k, &s));
    }
    files.insert("t-metatool".into(), recs);

    // T-clinc: in-scope queries with their intent and 6 random intents, out-of-scope ones with 7
    let cl = of("clinc");
    let pool = tool_set(&cl);
    let (ins, oos): (Vec<&Kept>, Vec<&Kept>) = cl.iter().partition(|k| k.gold.is_some());
    let n_oos = ((TEST_SLATES as f64) * NONE_SHARE).round() as usize;
    let mut recs = Vec::new();
    for (part, n, label) in [(&ins, TEST_SLATES - n_oos, "in"), (&oos, n_oos, "oos")] {
        for i in slates::draw(
            part.len(),
            n,
            &mut cx.rng(&format!("route/t-clinc/{label}")),
        ) {
            let k = part[i];
            let mut rng = cx.rng(&format!("route/t-clinc/slate/{}", k.r.id));
            let s = slates::test_slate(k.gold, &[], &pool, false, &mut rng, &|t| Some(t) != k.gold);
            recs.push(cx.record("t-clinc", "test", k.r.id.clone(), "test", k, &s));
        }
    }
    files.insert("t-clinc".into(), recs);

    // the sources whose items carry their own tool lists: every tool, then none
    for (source, name) in [
        ("tool-routing", "t-toolrouting"),
        ("bfcl-multiple", "t-bfcl"),
        ("bfcl-irrelevance", "t-bfcl"),
        ("when2call-mcq", "t-when2call"),
    ] {
        let mut recs = files.remove(name).unwrap_or_default();
        for k in of(source) {
            let mut rng = cx.rng(&format!("route/{name}/slate/{}", k.r.id));
            let mut tools = k.own.clone();
            rng.shuffle(&mut tools);
            let s = Slate::new(tools, k.gold);
            if s.tools.is_empty() {
                continue;
            }
            recs.push(cx.record(
                &format!("t-{source}"),
                "test",
                k.r.id.clone(),
                "test",
                k,
                &s,
            ));
        }
        files.insert(name.into(), recs);
    }

    // MASSIVE held-out scenarios, test partition: the intent, 6 random MASSIVE intents, none;
    // 20% removed; at most 1,000 per language
    let ms = of("massive-heldout");
    let mut pool: Vec<usize> = kept
        .iter()
        .filter(|k| k.r.source == "massive" || k.r.source == "massive-heldout")
        .filter_map(|k| k.gold)
        .collect();
    pool.sort_unstable();
    pool.dedup();
    let mut recs = Vec::new();
    for lang in ["en", "fr"] {
        let part: Vec<&Kept> = ms.iter().copied().filter(|k| k.r.lang == lang).collect();
        let drawn = slates::draw(
            part.len(),
            MASSIVE_PROBE,
            &mut cx.rng(&format!("route/t-massive/{lang}")),
        );
        let removed = slates::removed_set(
            drawn.len(),
            REMOVED_SHARE,
            &mut cx.rng(&format!("route/t-massive/{lang}/removed")),
        );
        for (j, &i) in drawn.iter().enumerate() {
            let k = part[i];
            let mut rng = cx.rng(&format!("route/t-massive/slate/{}", k.r.id));
            let s = slates::test_slate(k.gold, &[], &pool, removed.contains(&j), &mut rng, &|t| {
                Some(t) != k.gold
            });
            let kind = if removed.contains(&j) {
                "removed"
            } else {
                "test"
            };
            recs.push(cx.record(
                &format!("t-massive-{lang}"),
                "test",
                k.r.id.clone(),
                kind,
                k,
                &s,
            ));
        }
    }
    files.insert("t-massive".into(), recs);
    Ok(())
}

fn stage_of(cx: &Ctx) -> PathBuf {
    cx.l.stage.clone()
}

// ------------------------------------------------------------------------------------ slates

pub struct SlatesOptions {
    pub out: PathBuf,
    pub seed: u64,
    /// Without the teacher's answers: the hard slates unfiltered and the renamed gold with its
    /// own description. Such a build is marked provisional and is not trained on (smoke runs and
    /// rehearsals only).
    pub provisional: bool,
}

/// Step 5: the records of the four training kinds, for the training and validation requests.
pub fn build_slates(o: &SlatesOptions) -> Result<Value> {
    let st = stage(&o.out);
    let tools: Vec<Tool> = read_lines(&st.join("tools.jsonl"))?;
    let index: HashMap<String, usize> = tools
        .iter()
        .enumerate()
        .map(|(i, t)| (t.id.clone(), i))
        .collect();
    let plans: Vec<Plan> = read_lines(&st.join("plan.jsonl"))?;
    let able: HashMap<String, Vec<usize>> =
        read_or_empty::<Able>(&answers_path(&o.out, "false-negatives"))?
            .into_iter()
            .map(|a| (a.id, a.able))
            .collect();
    let rewrite: HashMap<String, String> =
        read_or_empty::<Description>(&answers_path(&o.out, "rewrite-description"))?
            .into_iter()
            .map(|d| (d.id, d.description))
            .collect();
    let refused: HashSet<String> = ["false-negatives", "rewrite-description"]
        .iter()
        .flat_map(|j| {
            let p = crate::distill::refused_path(&answers_path(&o.out, j));
            read_or_empty::<crate::distill::Refused>(&p).unwrap_or_default()
        })
        .map(|r| r.id)
        .collect();
    let mut counts = Counts::new();
    let fn_ids: HashMap<&str, String> = plans
        .iter()
        .map(|p| (p.id.as_str(), fn_id(p, &tools, &index)))
        .collect();
    let missing_fn = plans
        .iter()
        .filter(|p| {
            !able.contains_key(&fn_ids[p.id.as_str()]) && !refused.contains(&fn_ids[p.id.as_str()])
        })
        .count();
    let missing_rw = plans
        .iter()
        .filter_map(|p| p.gold.as_ref())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter(|g| !rewrite.contains_key(*g) && !refused.contains(*g))
        .count();
    if !o.provisional && (missing_fn > 0 || missing_rw > 0) {
        bail!(
            "the teacher has not answered every item ({missing_fn} false-negative items, {missing_rw} rewrites missing): \
             run scripts/route-teacher.sh, or build --provisional (not trained on)"
        );
    }
    let text = |t: usize| tools[t].option();
    let rec = |p: &Plan, s: &Slate, kind: &str| -> RouteRecord {
        let (opts, target) = s.render(text);
        RouteRecord {
            src: format!("{}:{}", p.group, p.source),
            t: "choice".into(),
            ins: INSTRUCTION.into(),
            // amendment 1: the request, then the BM25 score and rank of each tool option
            state: sky_infer::route::state_for_options(&p.user, &opts),
            opts,
            target,
            split: p.split.clone(),
            uid: format!("{}/{kind}", p.id),
            kind: kind.into(),
            lang: p.lang.clone(),
            request: p.id.clone(),
            tools: s.tools.iter().map(|&t| tools[t].id.clone()).collect(),
            follow: None,
        }
    };
    // per (group, split): the slates, those with a tool gold, the removed ones
    #[derive(Default)]
    struct Bucket {
        fixed: Vec<RouteRecord>,
        removed: Vec<RouteRecord>,
        spare: Vec<RouteRecord>,
        nocall: Vec<RouteRecord>,
    }
    let mut buckets: BTreeMap<(String, String), Bucket> = BTreeMap::new();
    let mut agree: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut alt: BTreeMap<(String, String), usize> = BTreeMap::new();
    for p in &plans {
        let ix = |v: &[String]| v.iter().map(|id| index[id]).collect::<Vec<_>>();
        let gold = p.gold.as_ref().map(|g| index[g]);
        let natural = Slate::new(ix(&p.natural), gold);
        let mut hard = Slate::new(ix(&p.hard), gold);
        // 4.5: the false-negative filter of the hard slate
        if let Some(a) = able.get(&fn_ids[p.id.as_str()]) {
            let mut reserve = ix(&p.reserve).into_iter();
            if let Some(gp) = hard.gold {
                let e = agree.entry(p.source.clone()).or_default();
                e.0 += 1;
                if !a.contains(&(gp + 1)) {
                    bump(
                        &mut counts,
                        format!(
                            "{} {}: dropped, the teacher says the gold cannot",
                            p.group, p.split
                        ),
                    );
                    continue;
                }
                e.1 += 1;
            }
            for (i, t) in hard.tools.clone().iter().enumerate() {
                if Some(i) == hard.gold || !a.contains(&(i + 1)) {
                    continue;
                }
                bump(
                    &mut counts,
                    format!("{} {}: distractors replaced", p.group, p.split),
                );
                let next = reserve.by_ref().find(|r| !hard.tools.contains(r) && r != t);
                match next {
                    Some(r) => hard.tools[i] = r,
                    None => bump(
                        &mut counts,
                        format!(
                            "{} {}: no tool left to replace a distractor",
                            p.group, p.split
                        ),
                    ),
                }
            }
        } else if !o.provisional {
            bump(
                &mut counts,
                format!(
                    "{} {}: refused by moderation, hard slate unfiltered",
                    p.group, p.split
                ),
            );
        }
        let b = buckets
            .entry((p.group.clone(), p.split.clone()))
            .or_default();
        match gold {
            Some(g) => {
                b.fixed.push(rec(p, &natural, "natural"));
                b.fixed.push(rec(p, &hard, "hard"));
                // removed: slate 1 or 2 alternately, without the gold
                let n = alt.entry((p.group.clone(), p.split.clone())).or_default();
                let (first, second) = if (*n).is_multiple_of(2) {
                    (&natural, &hard)
                } else {
                    (&hard, &natural)
                };
                *n += 1;
                for (s, spare) in [(first, false), (second, true)] {
                    let r = s.without_gold();
                    if r.tools.is_empty() {
                        bump(
                            &mut counts,
                            format!("{} {}: no removed slate (a gold alone)", p.group, p.split),
                        );
                        continue;
                    }
                    let r = rec(p, &r, "removed");
                    if spare {
                        b.spare.push(r)
                    } else {
                        b.removed.push(r)
                    }
                }
                // renamed (amendment 1): slate 1 with every tool named `tool_<k>`, distinct
                // random k; the gold's description is the teacher's rewrite (4.4), the others
                // are untouched
                let mut rng = Rng::derived(o.seed, &format!("route/renamed/{}", p.id));
                let gi = natural.gold.unwrap();
                let rewritten = rewrite
                    .get(&tools[g].id)
                    .cloned()
                    .map(|d| super::cut_description(&d))
                    .unwrap_or_else(|| tools[g].desc.clone());
                let s = slates::masked(
                    &natural,
                    |t| {
                        if t == natural.tools[gi] {
                            rewritten.clone()
                        } else {
                            tools[t].desc.clone()
                        }
                    },
                    &mut rng,
                );
                b.fixed.push(rec(p, &s, "renamed"));
            }
            None => {
                // no call (When2Call): natural and hard, gold none
                b.nocall.push(rec(p, &natural, "natural"));
                b.nocall.push(rec(p, &hard, "hard"));
            }
        }
    }
    // 4.4: none is the gold of 20% of the slates of each group and split
    let mut written = BTreeMap::new();
    let mut shares = BTreeMap::new();
    for ((group, split), mut b) in buckets {
        let t = b.fixed.len();
        let target = ((t as f64) * NONE_SHARE / (1.0 - NONE_SHARE)).round() as usize;
        let mut rng = Rng::derived(o.seed, &format!("route/none-share/{group}/{split}"));
        let mut none: Vec<RouteRecord> = Vec::new();
        if b.nocall.len() >= target {
            // more none-gold slates of requests without a call than the share allows
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
        let mut all = b.fixed;
        all.extend(none);
        all.sort_by(|a, b| a.uid.cmp(&b.uid));
        let n_none = all.iter().filter(|r| r.target.last() == Some(&1.0)).count();
        shares.insert(format!("{group} {split}"), json!({"slates": all.len(), "none gold": n_none, "share": n_none as f64 / all.len().max(1) as f64}));
        let path = o
            .out
            .join("records")
            .join(&group)
            .join(format!("{split}.jsonl"));
        written.insert(format!("{group}/{split}"), write_jsonl(&path, &all)?);
    }
    let agreement: BTreeMap<String, Value> = agree
        .iter()
        .map(|(s, (n, a))| (s.clone(), json!({"labelled": n, "gold named able": a, "rate": *a as f64 / (*n).max(1) as f64})))
        .collect();
    let report = json!({
        "provisional": o.provisional,
        "teacher answers missing": {"false-negatives": missing_fn, "rewrite-description": missing_rw},
        "records": written,
        "none share": shares,
        "teacher agreement with the gold (per source)": agreement,
        "counts": counts,
    });
    write_json(&o.out.join("counts/slates.json"), &report)?;
    Ok(report)
}

// ------------------------------------------------------------------------------------ render

/// The model's copy of a slate file (amendment 1): each record's state becomes the request and
/// the BM25 line of its tool options (`sky_infer::route::state_for_options`); nothing else
/// changes. The slate files themselves keep the request alone, as the baselines read it.
pub fn render(input: &Path, output: &Path) -> Result<Value> {
    let recs: Vec<Value> = read_lines(input)?;
    let mut out = Vec::with_capacity(recs.len());
    for mut r in recs {
        let state = r["state"].as_str().unwrap_or("");
        let Some(user) = state.strip_prefix("User: ") else {
            bail!("{}: a state that is not `User: <request>`", input.display());
        };
        if user.contains(&format!("\n{}", sky_infer::route::LINE_PREFIX)) {
            bail!("{}: already rendered", input.display());
        }
        let opts: Vec<String> = serde_json::from_value(r["opts"].clone())?;
        r["state"] = json!(sky_infer::route::state_for_options(user, &opts));
        out.push(r);
    }
    let n = write_jsonl(output, &out)?;
    Ok(json!({"rendered": n, "out": output.file_name().map(|f| f.to_string_lossy().to_string())}))
}

// ---------------------------------------------------------------------------------- manifest

pub struct ManifestOptions {
    pub data: PathBuf,
    pub out: PathBuf,
    pub seed: u64,
    /// `sky-train s2-truncation --json` reports, one per size and head window
    /// (`<size>:<window>=<path>`; each size with its own config and tokenizer).
    pub truncation: Vec<(String, usize, PathBuf)>,
}

fn hash_tree(root: &Path, rel: &str, out: &mut BTreeMap<String, Value>) -> Result<()> {
    let dir = root.join(rel);
    if !dir.exists() {
        return Ok(());
    }
    let mut entries: Vec<_> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    entries.sort();
    for p in entries {
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        let r = if rel.is_empty() {
            name.clone()
        } else {
            format!("{rel}/{name}")
        };
        if p.is_dir() {
            hash_tree(root, &r, out)?;
        } else {
            out.insert(r, crate::secrets2::file_entry(&p)?);
        }
    }
    Ok(())
}

/// The head window of section 3: the smallest of 160, 256, 320 at which at most 5% of the
/// records have an option shrunk (the `head` share of `s2-truncation` over the training and
/// validation records); `None` when no candidate meets the rule.
pub fn head_window(reports: &[(usize, Value)]) -> Option<usize> {
    let mut v: Vec<&(usize, Value)> = reports.iter().collect();
    v.sort_by_key(|(w, _)| *w);
    v.into_iter()
        .find(|(_, r)| r["all"]["head"].as_f64().is_some_and(|h| h <= 0.05))
        .map(|(w, _)| *w)
}

/// Step 6: `<out>/BUILD.json`.
pub fn manifest(o: &ManifestOptions) -> Result<Value> {
    let mut outputs = BTreeMap::new();
    for rel in [
        "records", "val", "test", "stage", "teacher", "emb", "counts",
    ] {
        hash_tree(&o.out, rel, &mut outputs)?;
    }
    let mut inputs = BTreeMap::new();
    for kind in ["src", "models"] {
        let dir = o.data.join(kind);
        if !dir.exists() {
            continue;
        }
        let mut names: Vec<_> = std::fs::read_dir(&dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .collect();
        names.sort();
        for d in names {
            let m = d.join(crate::fetch::MANIFEST);
            if m.exists() {
                let v = super::read_json(&m)?;
                inputs.insert(format!("{kind}/{}", d.file_name().unwrap().to_string_lossy()), json!({
                    "repo": v["repo"], "kind": v["kind"], "commit": v["commit"],
                    "bytes": v["files"].as_array().map_or(0, |f| f.iter().filter_map(|x| x["size"].as_u64()).sum::<u64>()),
                    "manifest_sha256": sha256_file(&m)?,
                }));
            }
        }
    }
    let mut by_size: BTreeMap<String, Vec<(usize, Value)>> = BTreeMap::new();
    for (size, w, p) in &o.truncation {
        by_size
            .entry(size.clone())
            .or_default()
            .push((*w, super::read_json(p)?));
    }
    let window: BTreeMap<String, Option<usize>> = by_size
        .iter()
        .map(|(s, t)| (s.clone(), head_window(t)))
        .collect();
    let reports: BTreeMap<String, Vec<Value>> = by_size
        .iter()
        .map(|(s, t)| {
            (
                s.clone(),
                t.iter()
                    .map(|(w, r)| json!({"window": w, "all": r["all"]}))
                    .collect(),
            )
        })
        .collect();
    let counts: BTreeMap<String, Value> = ["requests", "split", "slates"]
        .iter()
        .filter_map(|k| {
            super::read_json(&o.out.join(format!("counts/{k}.json")))
                .ok()
                .map(|v| (k.to_string(), v))
        })
        .collect();
    let provisional = counts
        .get("slates")
        .and_then(|v| v["provisional"].as_bool())
        .unwrap_or(true);
    let test: BTreeMap<String, Value> = outputs
        .iter()
        .filter(|(k, _)| k.starts_with("test/"))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let m = json!({
        "about": "s6-route-nano (docs/s6-nano2-prereg.md, section 4): the data build",
        "seed": o.seed,
        "provisional": provisional,
        "inputs": inputs,
        "test files (hashed for the amendment of 4.6, item 1; never read before section 7)": test,
        "truncation": {
            "rule": "the smallest of 160, 256, 320 at which at most 5% of the training and validation records have an option shrunk (s2-truncation `head`)",
            "reports": reports,
            "window": window,
        },
        "counts": counts,
        "outputs": outputs,
    });
    write_json(&o.out.join("BUILD.json"), &m)?;
    Ok(
        json!({"written": "BUILD.json", "provisional": provisional, "window": window, "test files": test.len()}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route::slates::TEST_TOOLS;

    /// A `.npy` file of float32 rows.
    fn write_npy(path: &Path, rows: &[Vec<f32>]) {
        let header = format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}",
            rows.len(),
            rows[0].len()
        );
        let mut h = header;
        while !(10 + h.len() + 1).is_multiple_of(64) {
            h.push(' ');
        }
        h.push('\n');
        let mut buf = b"\x93NUMPY\x01\x00".to_vec();
        buf.extend_from_slice(&(h.len() as u16).to_le_bytes());
        buf.extend_from_slice(h.as_bytes());
        for r in rows {
            for x in r {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
        std::fs::write(path, buf).unwrap();
    }

    fn req(
        id: &str,
        group: &str,
        split: Option<&str>,
        user: &str,
        tools: &[&Tool],
        gold: Option<&Tool>,
    ) -> Request {
        Request {
            id: id.into(),
            source: id.split(':').next().unwrap().into(),
            group: group.into(),
            split: split.map(str::to_string),
            lang: "en".into(),
            user: user.into(),
            tools: tools.iter().map(|t| t.id.clone()).collect(),
            gold: gold.map(|t| t.id.clone()),
            exclude: Vec::new(),
            kind: if gold.is_some() {
                "call"
            } else {
                "cannot_answer"
            }
            .into(),
        }
    }

    /// An invented build: 400 tools in 200 near-duplicate pairs, 50 requests per pair (over
    /// the cap), When2Call requests with and without a gold, intents.
    fn fixture(dir: &Path) {
        let st = stage(dir);
        std::fs::create_dir_all(&st).unwrap();
        std::fs::create_dir_all(dir.join("emb")).unwrap();
        let mut rng = Rng::new(42);
        let mut tools = Vec::new();
        let mut vec_of: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        for i in 0..200 {
            let base: Vec<f32> = (0..16).map(|_| rng.f64() as f32 - 0.5).collect();
            for v in 0..2 {
                let t = Tool::new(
                    &format!("get_thing{i}_v{v}"),
                    &format!("Does thing {i}, way {v}."),
                );
                let e: Vec<f32> = base.iter().map(|x| x + 0.01 * v as f32).collect();
                vec_of.insert(t.id.clone(), e);
                tools.push(t);
            }
        }
        let intents: Vec<Tool> = (0..6)
            .map(|i| Tool::new(&format!("intent_{i}"), &format!("The user wants {i}.")))
            .collect();
        for t in &intents {
            vec_of.insert(
                t.id.clone(),
                (0..16).map(|_| rng.f64() as f32 - 0.5).collect(),
            );
        }
        let mut reqs = Vec::new();
        for i in 0..200 {
            let (a, b) = (&tools[2 * i], &tools[2 * i + 1]);
            let other = &tools[(2 * i + 7) % 400];
            for j in 0..50 {
                reqs.push(req(
                    &format!("glaive:{i}-{j}"),
                    TOOLS,
                    None,
                    &format!("please do thing {i} now {j}"),
                    &[a, b, other],
                    Some(if j % 2 == 0 { a } else { b }),
                ));
            }
        }
        for j in 0..40 {
            let g = &tools[(j * 11) % 400];
            let own = [g, &tools[(j * 11 + 3) % 400]];
            reqs.push(req(
                &format!("when2call:{j}"),
                WHEN2CALL,
                Some(if j % 5 == 0 { "val" } else { "train" }),
                &format!("w2c call {j}"),
                &own,
                Some(g),
            ));
            reqs.push(req(
                &format!("when2call:n{j}"),
                WHEN2CALL,
                Some("train"),
                &format!("w2c none {j}"),
                &own[1..],
                None,
            ));
        }
        for j in 0..60 {
            let g = &intents[j % 6];
            reqs.push(req(
                &format!("massive:{j}"),
                INTENTS,
                Some(if j % 6 == 0 { "val" } else { "train" }),
                &format!("intent text {j}"),
                &[],
                Some(g),
            ));
        }
        let mut all: Vec<Tool> = tools.iter().chain(intents.iter()).cloned().collect();
        all.sort();
        write_jsonl(&st.join("tools.jsonl"), &all).unwrap();
        write_jsonl(&st.join("requests.jsonl"), &reqs).unwrap();
        write_json(&st.join("similar.json"), &json!({})).unwrap();
        write_jsonl::<String>(&st.join("metatool-tools.jsonl"), []).unwrap();
        let texts: Vec<Value> = vec_of
            .keys()
            .map(|id| json!({"id": id, "text": all.iter().find(|t| &t.id == id).unwrap().option()}))
            .collect();
        write_jsonl(&st.join("texts.jsonl"), &texts).unwrap();
        let rows: Vec<Vec<f32>> = vec_of.values().cloned().collect();
        write_npy(&dir.join("emb/minilm.npy"), &rows);
        write_npy(&dir.join("emb/potion.npy"), &rows);
        write_json(
            &dir.join("emb/meta.json"),
            &json!({"texts_sha256": sha256_file(&st.join("texts.jsonl")).unwrap()}),
        )
        .unwrap();
    }

    fn hashes(dir: &Path) -> BTreeMap<String, Value> {
        let mut h = BTreeMap::new();
        for rel in ["records", "val", "test", "stage", "teacher"] {
            hash_tree(dir, rel, &mut h).unwrap();
        }
        h
    }

    fn run(dir: &Path, seed: u64) {
        split(&SplitOptions {
            out: dir.into(),
            seed,
        })
        .unwrap();
        build_slates(&SlatesOptions {
            out: dir.into(),
            seed,
            provisional: true,
        })
        .unwrap();
    }

    #[test]
    fn an_invented_build() {
        let root = std::env::temp_dir().join(format!("sky-route-build-{}", std::process::id()));
        let (a, b, c) = (root.join("a"), root.join("b"), root.join("c"));
        for d in [&a, &b, &c] {
            fixture(d);
        }
        run(&a, 7);
        run(&b, 7);
        run(&c, 8);
        // determinism under the seed, and the seed matters
        assert_eq!(hashes(&a), hashes(&b));
        assert_ne!(
            hashes(&a)["records/tools/train.jsonl"],
            hashes(&c)["records/tools/train.jsonl"]
        );

        let tools: Vec<Tool> = read_lines(&stage(&a).join("tools.jsonl")).unwrap();
        let split_report = super::super::read_json(&a.join("counts/split.json")).unwrap();
        assert!(split_report["clusters by split"]["test"].as_u64().unwrap() > 0);
        // the clusters: the two variants of a thing are one cluster, so one split
        let side = |recs: &[RouteRecord]| -> BTreeSet<String> {
            recs.iter()
                .flat_map(|r| {
                    r.tools
                        .iter()
                        .map(|id| tools.iter().find(|t| &t.id == id).unwrap().name.clone())
                })
                .map(|n| n.split("_v").next().unwrap().to_string())
                .collect()
        };
        let train: Vec<RouteRecord> = read_lines(&a.join("records/tools/train.jsonl")).unwrap();
        let w2c: Vec<RouteRecord> = read_lines(&a.join("records/when2call/train.jsonl")).unwrap();
        let test: Vec<RouteRecord> = read_lines(&a.join("test/t_tools.jsonl")).unwrap();
        let val: Vec<RouteRecord> = read_lines(&a.join("val/v_tools.jsonl")).unwrap();
        let (tr, te, va) = (
            side(&[train.clone(), w2c].concat()),
            side(&test),
            side(&val),
        );
        // no test cluster in training, and validation apart from both
        assert!(
            tr.is_disjoint(&te),
            "{:?}",
            tr.intersection(&te).collect::<Vec<_>>()
        );
        assert!(tr.is_disjoint(&va) && va.is_disjoint(&te));
        // the per-cluster cap: 30 training requests per thing at most
        let mut per: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for r in &train {
            let g = r.target.iter().position(|&x| x == 1.0).unwrap();
            if g < r.tools.len() {
                let name = tools
                    .iter()
                    .find(|t| t.id == r.tools[g])
                    .unwrap()
                    .name
                    .split("_v")
                    .next()
                    .unwrap()
                    .to_string();
                per.entry(name).or_default().insert(r.request.clone());
            }
        }
        assert!(
            per.values().all(|s| s.len() <= TRAIN_CAP),
            "{:?}",
            per.values().map(BTreeSet::len).max()
        );
        assert!(per.values().any(|s| s.len() == TRAIN_CAP));
        // the four kinds, and none last everywhere
        let kinds: BTreeSet<&str> = train.iter().map(|r| r.kind.as_str()).collect();
        assert_eq!(
            kinds,
            BTreeSet::from(["hard", "natural", "removed", "renamed"])
        );
        for r in train.iter().chain(&test) {
            assert_eq!(r.opts.last().unwrap(), super::super::NONE_OPTION);
            assert!((2..=8).contains(&r.opts.len()));
            assert_eq!(r.target.iter().sum::<f32>(), 1.0);
            assert_eq!(r.ins, INSTRUCTION);
            assert!(r.state.starts_with("User: "));
        }
        // amendment 1: a renamed slate renames every tool, with distinct numbers; the
        // distractors keep their descriptions
        let by_id: BTreeMap<&str, &Tool> = tools.iter().map(|t| (t.id.as_str(), t)).collect();
        for r in train.iter().filter(|r| r.kind == "renamed") {
            let names: BTreeSet<&str> = r.opts[..r.opts.len() - 1]
                .iter()
                .map(|o| sky_infer::route::option_name(o))
                .collect();
            assert_eq!(names.len(), r.opts.len() - 1, "{r:?}");
            assert!(names.iter().all(|n| n.starts_with("tool_")));
            let g = r.target.iter().position(|&x| x == 1.0).unwrap();
            for (i, (o, id)) in r.opts.iter().zip(&r.tools).enumerate() {
                if i != g {
                    assert!(
                        o.ends_with(&format!(": {}", by_id[id.as_str()].desc)),
                        "{o}"
                    );
                }
            }
        }
        // amendment 1: the training states carry the BM25 line of their tool options; the
        // test and validation slates do not (the model's copy is rendered at evaluation)
        for r in &train {
            assert_eq!(
                r.state,
                sky_infer::route::state_for_options(&r.state.lines().next().unwrap()[6..], &r.opts)
            );
            assert_eq!(r.state.contains("\nBM25: "), r.opts.len() > 1);
        }
        assert!(test.iter().all(|r| !r.state.contains('\n')));
        let rendered = root.join("rendered.jsonl");
        render(&a.join("val/v_tools.jsonl"), &rendered).unwrap();
        let rv: Vec<RouteRecord> = read_lines(&rendered).unwrap();
        assert_eq!(rv.len(), val.len());
        for (x, y) in rv.iter().zip(&val) {
            assert_eq!(x.opts, y.opts);
            assert_eq!(
                x.state,
                sky_infer::route::state_for_options(&y.state[6..], &y.opts)
            );
        }
        assert!(render(&rendered, &root.join("twice.jsonl")).is_err());
        // the none share of every group and split
        let report = super::super::read_json(&a.join("counts/slates.json")).unwrap();
        for (k, v) in report["none share"].as_object().unwrap() {
            let (n, none) = (
                v["slates"].as_u64().unwrap(),
                v["none gold"].as_u64().unwrap(),
            );
            let want = ((n - none) as f64 * NONE_SHARE / (1.0 - NONE_SHARE)).round() as u64;
            assert_eq!(none, want, "{k}: {v}");
        }
        // the test slates: 8 options, 20% removed
        assert!(test.iter().all(|r| r.opts.len() == TEST_TOOLS + 1));
        let removed = test.iter().filter(|r| r.kind == "removed").count();
        assert_eq!(
            removed,
            (test.len() as f64 * REMOVED_SHARE).round() as usize
        );
        assert!(test
            .iter()
            .filter(|r| r.kind == "removed")
            .all(|r| r.target[TEST_TOOLS] == 1.0));
        // a test file is written once: one that differs is refused
        let p = a.join("test/t_tools.jsonl");
        let mut bytes = std::fs::read(&p).unwrap();
        bytes.push(b'\n');
        std::fs::write(&p, bytes).unwrap();
        assert!(split(&SplitOptions {
            out: a.clone(),
            seed: 7
        })
        .is_err());
        // without the teacher's answers, only a provisional build
        assert!(build_slates(&SlatesOptions {
            out: b.clone(),
            seed: 7,
            provisional: false
        })
        .is_err());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn the_head_window_rule() {
        let r = |h: f64| json!({"all": {"head": h}});
        assert_eq!(
            head_window(&[(320, r(0.0)), (160, r(0.2)), (256, r(0.05))]),
            Some(256)
        );
        assert_eq!(head_window(&[(160, r(0.01)), (256, r(0.0))]), Some(160));
        assert_eq!(
            head_window(&[(160, r(0.3)), (256, r(0.2)), (320, r(0.06))]),
            None
        );
    }
}
